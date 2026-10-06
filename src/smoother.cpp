#include <cassert>
#include <random>

#include "smoother.hpp"


namespace estimation {

    // =========================================================================
    // Exact Rauch-Tung-Striebel smoother on a forward record
    // =========================================================================
    //
    // Included as the independent reference that the Bryson-Frazier EnKS is
    // checked against.  Note that RTS needs P_pred^{-1}, the N x N forecast
    // covariance inverse -- which is exactly why the EnKS below does NOT use
    // this form.  See the EnKS documentation.

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
            Matrix P_pred = (F[i] * record.P_posterior[i] * F[i].transpose()
                             + Q[i]).eval();
            P_pred = ((P_pred + P_pred.transpose()).eval() * T(0.5));
            Vector x_pred = (F[i] * record.x_posterior[i]).eval();
            if (!u.empty()) x_pred += u[i];

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
    // EnKS: the localized ensemble Kalman smoother
    // =========================================================================
    //
    // Butala, Fathpour & Bhatt, "A Localized Ensemble Kalman Smoother" (IEEE,
    // 2012): a Monte Carlo approximation to the BRYSON-FRAZIER smoother.  Three
    // stages, of which the second is what makes it usable at L < N:
    //
    //  1. Run the forward filter (here the EnKF), storing at each i
    //        x~_{i|i}    the filtered mean
    //        P~_{i|i}    the posterior sample covariance
    //        K~_i        the gain (tapered if a taper is set)
    //        e~_i = y_i - H_i x~_{i|i-1}                  (innovation)
    //        R~_{e,i} = R_i + H_i (C_i o P~_{i|i-1}) H_i^T   (innovation covariance)
    //
    //  2. Backward recursion on an ENSEMBLE adjoint variable (eqs. 12-13),
    //     with Z_i columns iid N(0, I):
    //        lam_{I+1} = 0 ,  Lam_{I+1} = 0
    //        lam_i   = (I - K~_i H_i)^T F_i^T lam_{i+1} + H_i^T R~_{e,i}^{-1} e~_i
    //        Lam_i   = (I - K~_i H_i)^T F_i^T Lam_{i+1} + H_i^T R~_{e,i}^{-1/2} Z_i
    //     so that E[Lam_i Lam_i^T/(L-1)] = Lambda_i, the exact recursion (8).
    //     (The H_i^T on the noise term is required for that identity; the
    //     journal equation as extracted drops it.)
    //
    //  3. Smoothed estimates (eqs. 9-10 / 16-17):
    //        x~_{i|1:I} = x~_{i|i} + (C'_i o P~_{i|i}) F_i^T lam_{i+1}
    //        P~_{i|1:I} = P~_{i|i} - (C'_i o P~_{i|i}) F_i^T (C''_i o Lam_{i+1})
    //                              F_i (C'_i o P~_{i|i})
    //
    // THE ONLY INVERSE APPEARING ANYWHERE IS R~_{e,i}^{-1}:  an M x M innovation
    // covariance, with R~_{e,i} >= R_i > 0, hence always invertible and
    // independent of L and N.  That is the entire reason for using the
    // Bryson-Frazier rather than the Rauch-Tung-Striebel form: RTS requires
    // P_{i|i-1}^{-1} (N x N, singular whenever L-1 < N), and the
    // Mayne-Fraser-Potter form requires F_i^{-1} (false for most physical
    // models, which lose information going forward).  The smoothed estimates
    // are therefore available for any L and any N.

    template <typename T>
    EnKS<T>::EnKS(const std::size_t N, const std::size_t L,
                  const std::uint64_t seed)
        : fwd_(N, L, seed), rng_(seed ? seed : 0x9e3779b97f4a7c15ULL)
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
        using Matrix = typename Types<T>::Matrix;
        using Vector = typename Types<T>::Vector;

        const std::size_t I = y.size();
        const Eigen::Index L = static_cast<Eigen::Index>(this->fwd_.ensemble_size());
        const Scalar denom = Scalar(L - 1);
        require(I > 0, "smooth: empty sequence");
        require(H.size() == I && R.size() == I && F.size() == I && Q.size() == I,
                "smooth: y, H, R, F, Q must all have the same length");
        require(u.empty() || u.size() == I,
                "smooth: u must be empty or match the sequence length");

        // ---- Stage 1: forward filter, storing what the adjoint needs -------
        std::vector<Vector> x_post(I), e(I);
        std::vector<Matrix> P_post(I), Ki(I), Ai(I), Re_chol(I);

        for (std::size_t i = 0; i < I; ++i) {
            const Vector x_pred = this->fwd_.state();
            const Matrix P_pred = this->fwd_.covariance();
            const Eigen::Index m = H[i].rows();
            const Eigen::Index n = static_cast<Eigen::Index>(this->fwd_.dimension());

            // Gain and innovation covariance of the (possibly tapered) prior.
            const Matrix Pg   = this->fwd_.tapered(P_pred);
            const Matrix P_HT = Pg * H[i].transpose();
            Matrix S = (H[i] * P_HT + R[i]).eval();
            S = ((S + S.transpose()).eval() * Scalar(0.5));
            Eigen::LLT<Matrix> lltS(S);
            require(lltS.info() == Eigen::Success,
                    "smooth: innovation covariance is not positive definite");
            const Matrix K = lltS.solve(P_HT.transpose()).transpose();

            e[i]  = (y[i] - H[i] * x_pred).eval();          // eq. (11)
            Ki[i] = K;
            Ai[i] = ((Matrix::Identity(n, n) - K * H[i]).transpose()
                     * F[i].transpose()).eval();            // (I - K H)^T F^T
            // The Cholesky factor of R~_e is the only factorization needed; its
            // inverse is never formed.
            Re_chol[i] = lltS.matrixL();

            this->fwd_.measurement_update(y[i], H[i], R[i]);
            x_post[i] = this->fwd_.state();
            P_post[i] = this->fwd_.covariance();
            if (!u.empty()) this->fwd_.time_update(F[i], Q[i], u[i]);
            else            this->fwd_.time_update(F[i], Q[i]);
        }

        // ---- Stage 2: backward recursion on the ensemble adjoint -----------
        // lam is N x 1 (the adjoint mean), Lam is N x L (its ensemble).
        std::vector<Vector> lam(I + 1);
        std::vector<Matrix> Lam(I + 1);
        std::normal_distribution<double> gauss(0.0, 1.0);

        lam[I] = Vector::Zero(static_cast<Eigen::Index>(this->fwd_.dimension()));
        Lam[I] = Matrix::Zero(static_cast<Eigen::Index>(this->fwd_.dimension()), L);

        for (std::size_t k = I; k-- > 0;) {
            const Eigen::Index m = H[k].rows();
            // R~_e^{-1/2} Z, by triangular solves with the Cholesky factor of
            // R~_e -- the ONLY inversion in the algorithm, and it is M x M.
            Matrix Zm(m, L);
            for (Eigen::Index j = 0; j < L; ++j)
                for (Eigen::Index a = 0; a < m; ++a)
                    Zm(a, j) = static_cast<Scalar>(gauss(rng_));

            Matrix W(m, L);
            for (Eigen::Index j = 0; j < L; ++j) {
                W.col(j) = Re_chol[k]
                    .template triangularView<Eigen::Lower>().solve(Zm.col(j));
            }

            lam[k] = (Ai[k] * lam[k + 1]).eval();
            if (e[k].size() > 0) {
                Vector w(m);
                w = Re_chol[k].template triangularView<Eigen::Lower>().solve(e[k]);
                w = Re_chol[k].transpose()
                        .template triangularView<Eigen::Upper>().solve(w);
                lam[k] += (H[k].transpose() * w).eval();      // H^T R~_e^{-1} e
            }
            Lam[k] = (Ai[k] * Lam[k + 1] + (H[k].transpose() * W)).eval();
        }

        // ---- Stage 3: smoothed estimates ------------------------------------
        SmoothOutput<T> out;
        out.x_smoothed.assign(I, Vector());
        out.P_smoothed.assign(I, Matrix());
        for (std::size_t i = 0; i < I; ++i) {
            const Matrix Cp = this->fwd_.tapered(P_post[i]);     // C'_i o P~
            const Vector dx = (Cp * (F[i].transpose() * lam[i + 1])).eval();
            out.x_smoothed[i] = (x_post[i] + dx).eval();

            // P_{i|1:I} = P~_{i|i} - (C' o P~) F^T (C'' o Lam_{i+1}) F (C' o P~)
            // with Lam = Lam_tilde Lam_tilde^T/(L-1).  Kept in factored form.
            Matrix mid = (Lam[i + 1] * Lam[i + 1].transpose()) / denom;
            mid = this->fwd_.tapered(mid);                       // C''_i o Lam
            Matrix P = (P_post[i]
                        - (Cp * F[i].transpose() * mid * F[i] * Cp)).eval();
            P = ((P + P.transpose()).eval() * Scalar(0.5));
            out.P_smoothed[i] = P;
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
