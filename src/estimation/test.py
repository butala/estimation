import numpy as np

from estimation.lib import KF


if __name__ == '__main__':
    M = 6
    N = 10

    x_hat  = np.asfortranarray(np.random.randn(N))
    P_sqrt = np.random.randn(N, N)
    P      = np.asfortranarray(P_sqrt.T @ P_sqrt)
    y      = np.asfortranarray(np.random.randn(M))
    H      = np.asfortranarray(np.random.randn(M, N))
    R_sqrt = np.random.randn(M, M)
    R      = np.asfortranarray(R_sqrt.T @ R_sqrt)
    # F      = np.asfortranarray(np.random.randn(N, N))
    # Q_sqrt = np.random.randn(N, N)
    # Q      = np.asfortranarray(Q_sqrt.T @ Q_sqrt)
    F      = np.asfortranarray(np.eye(N))
    Q      = np.asfortranarray(np.zeros((N, N)))

    kf = KF(10)

    kf.initialize(x_hat, P)
    z = kf.measurement_update(y, H, R)
    print(z)
    print()
    print(kf.P())
    print()

    from pyrsss.kalman.kalman_filter import kalman_filter

    out = kalman_filter([y], [H], [R], [F], [Q], x_hat, P)

    print(out.x_hat[0])
    print()
    print(out.P[0])
    print()
