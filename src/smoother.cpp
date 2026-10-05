#include <cassert>

#include "smoother.hpp"


namespace estimation {

    // =========================================================================
    // Exact Rauch-Tung-Striebel smoother on a forward record
    // =========================================================================

    template <typename T>
    SmoothOutput<T>
    rts_smooth(const BatchOutput<T> &record,
               const std::vector<typename Types<T>::ConstMatrixRef> &F,
               const std::vector<typename Types<T>::ConstMatrixRef> &Q,
               const std::vector<typename Types<T>::ConstVectorRef> &u)
    {
        using Vector = typename Types<T>::Vector;
        using Matrix = typename Types<T>::Matrix;

        const std::size_t I = record.x_posterior.size();
        require(I > 0, "rts_smooth: empty record");
        require(record.x_prior.size() == I && record.P_prior.size() == I &&
                record.P_posterior.size() == I,
                "rts_smooth: record must contain means and covariances");
        require(F.size() == I && Q.size() == I,
                "rts_smooth: F and Q must match the record length");
        require(u.empty() || u.size() == I,
                "rts_smooth: u must be empty or match the record length");

        SmoothOutput<T> out;
        out.x_smoothed.assign(I, Vector());
        out.P_smoothed.assign(I, Matrix());

        out.x_smoothed[I - 1] = record.x_posterior[I - 1];
        out.P_smoothed[I - 1] = record.P_posterior[I - 1];

        for (std::size_t i = I - 1; i-- > 0;) {
            // One-step prediction from the analysis at i. Recomputed rather
            // than read from record.x_prior[i+1] so the routine also works on
            // chunked or trimmed records.
            Matrix P_pred = (F[i] * record.P_posterior[i] * F[i].transpose()
                             + Q[i]).eval();
            P_pred = ((P_pred + P_pred.transpose()).eval() * T(0.5));
            Vector x_pred = (F[i] * record.x_posterior[i]).eval();
            if (!u.empty()) x_pred += u[i];

            // G = P^a_i F^T (P_pred)^{-1}
            const Matrix PHT = (record.P_posterior[i] * F[i].transpose()).eval();
            Eigen::LLT<Matrix> llt(P_pred);
            require(llt.info() == Eigen::Success,
                    "rts_smooth: predicted covariance is not positive definite");
            const Matrix G = llt.solve(PHT.transpose()).transpose();

            const Vector dx = (out.x_smoothed[i + 1] - x_pred).eval();
            out.x_smoothed[i] = (record.x_posterior[i] + G * dx).eval();

            Matrix dP = (out.P_smoothed[i + 1] - P_pred).eval();
            out.P_smoothed[i] =
                (record.P_posterior[i] + G * dP * G.transpose()).eval();
            out.P_smoothed[i] =
                ((out.P_smoothed[i] + out.P_smoothed[i].transpose()).eval()
                 * T(0.5));
        }
        return out;
    }


    // =========================================================================
    // EnKS: Evensen's fixed-interval ensemble smoother
    // =========================================================================

    template <typename T>
    EnKS<T>::EnKS(const std::size_t N, const std::size_t L,
                  const std::uint64_t seed)
        : fwd_(N, L, seed)
    {
    }


    template <typename T>
    SmoothOutput<T>
    EnKS<T>::smooth(const std::vector<ConstVectorRef> &y,
                    const std::vector<ConstMatrixRef> &H,
                    const std::vector<ConstMatrixRef> &R,
                    const std::vector<ConstMatrixRef> &F,
                    const std::vector<ConstMatrixRef> &Q,
                    const std::vector<ConstVectorRef> &u)
    {
        const std::size_t I = y.size();
        const Scalar denom = Scalar(this->fwd_.ensemble_size() - 1);
        require(I > 0, "smooth: empty sequence");
        require(H.size() == I && R.size() == I && F.size() == I && Q.size() == I,
                "smooth: y, H, R, F, Q must all have the same length");
        require(u.empty() || u.size() == I,
                "smooth: u must be empty or match the sequence length");

        // ---- forward pass: store forecast and analysis ensembles ----------
        // Xf[i] / xf[i] is the ensemble entering measurement_update at time i
        // (so xf[i+1] already includes the input u[i]).
        std::vector<Matrix> Xf(I), Xa(I);
        std::vector<Vector> xf(I), xa(I);
        for (std::size_t i = 0; i < I; ++i) {
            Xf[i] = this->fwd_.anomalies();
            xf[i] = this->fwd_.state();
            this->fwd_.measurement_update(y[i], H[i], R[i]);
            Xa[i] = this->fwd_.anomalies();
            xa[i] = this->fwd_.state();
            if (!u.empty()) this->fwd_.time_update(F[i], Q[i], u[i]);
            else            this->fwd_.time_update(F[i], Q[i]);
        }

        // ---- backward pass with lag-one cross-covariances -----------------
        SmoothOutput<T> out;
        out.x_smoothed.assign(I, Vector());
        out.P_smoothed.assign(I, Matrix());
        std::vector<Matrix> Xs(I);

        out.x_smoothed[I - 1] = xa[I - 1];
        Xs[I - 1] = Xa[I - 1];

        for (std::size_t i = I - 1; i-- > 0;) {
            // C = Cov(x^a_i, x^f_{i+1}); tapered if a taper is set (LEKS).
            const Matrix C = this->fwd_.tapered(
                (Xa[i] * Xf[i + 1].transpose()) / denom);

            Matrix P_f = ((Xf[i + 1] * Xf[i + 1].transpose()) / denom);
            P_f = ((P_f + P_f.transpose()).eval() * Scalar(0.5));
            Eigen::LLT<Matrix> llt(P_f);
            require(llt.info() == Eigen::Success,
                    "smooth: forecast covariance is not positive definite");
            const Eigen::Index n = P_f.rows();
            const Matrix M = llt.solve(Matrix::Identity(n, n));   // P_f^{-1}

            // x^s_i = x^a_i + C P_f^{-1} (x^s_{i+1} - x^f_{i+1}), and the same
            // map applied to the anomalies (x^f_{i+1} already includes u[i]).
            const Vector dx = (out.x_smoothed[i + 1] - xf[i + 1]).eval();
            out.x_smoothed[i] = (xa[i] + (C * M) * dx).eval();
            const Matrix dX = (Xs[i + 1] - Xf[i + 1]).eval();
            Xs[i] = (Xa[i] + (C * M) * dX).eval();
        }

        for (std::size_t i = 0; i < I; ++i) {
            Matrix P = ((Xs[i] * Xs[i].transpose()) / denom);
            out.P_smoothed[i] = ((P + P.transpose()).eval() * Scalar(0.5));
        }
        return out;
    }


    template class EnKS<double>;
    template class EnKS<float>;
    template class LEKS<double>;
    template class LEKS<float>;

    template SmoothOutput<double>
    rts_smooth(const BatchOutput<double> &,
               const std::vector<Types<double>::ConstMatrixRef> &,
               const std::vector<Types<double>::ConstMatrixRef> &,
               const std::vector<Types<double>::ConstVectorRef> &);
    template SmoothOutput<float>
    rts_smooth(const BatchOutput<float> &,
               const std::vector<Types<float>::ConstMatrixRef> &,
               const std::vector<Types<float>::ConstMatrixRef> &,
               const std::vector<Types<float>::ConstVectorRef> &);

} // namespace estimation
