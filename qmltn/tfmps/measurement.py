import numpy as np
import tensorflow as tf
from qmltn.utils.mps import bond_dimension


class MPSMeasurement(tf.keras.layers.Layer):
    """MPS layer that is used for classification
    n:          number of classical parameters
    nout:       number of classes
    D:          Maximum bond dimension of the MPS
    d:          local Hilbert space dimension
    boundary:   determines the boundary condition for the MPS
    init:       Type of initial condition
    eps:        Each mps matrix is multiplied by eps
    use_biases: Determines if we use the biases
    normalize:  determines if we divide the output with its one norm
    init_diag:  Determines the initial values on the diagonal of the MPS
    ti:         Translationally invariant measurement
    """

    def __init__(
        self,
        nout,
        D=10,
        d=2,
        boundary="c",
        eps=1.0,
        use_biases=False,
        normalize=True,
        init_diag=1.0,
        name="L1_measurement",
        ti=False,
    ):
        super(MPSMeasurement, self).__init__(name=name)
        self.nout = nout
        self.D = D
        self.d = d
        self.boundary = boundary
        self.eps = eps
        self.use_biases = use_biases
        self.normalize = normalize
        self.init_diag = init_diag
        self.ti = ti

    def get_config(self):
        conf = {}
        conf["nout"] = self.nout
        conf["D"] = self.D
        conf["d"] = self.d
        conf["boundary"] = self.boundary
        conf["eps"] = self.eps
        conf["use_biases"] = self.use_biases
        conf["normalize"] = self.normalize
        conf["init_diag"] = self.init_diag
        conf["ti"] = self.ti
        conf["name"] = self.name
        return conf

    def get_mps(self, i):
        bias = None
        if self.ti:
            if i == 0:
                if self.use_biases:
                    bias = self.biases[0]
                return self.mps[0], bias
            if i == self.n-1:
                if self.use_biases:
                    bias = self.biases[2]
                return self.mps[2], bias
            if self.use_biases:
                bias = self.biases[1]
            return self.mps[1], bias
        else:
            if self.use_biases:
                bias = self.biases[i]
            return self.mps[i], bias

    def build(self, input_shape):
        """
        On the first run we construct the mps matrices.
        The shape of which is automatically determined from the input_shape
        """
        self.n = len(input_shape)
        self.iout = self.n // 2
        self.mps, self.biases, self.Aout = measurement_kernels(
            n=self.n,
            D=self.D,
            nout=self.nout,
            iout=self.iout,
            d=self.d,
            boundary=self.boundary,
            use_biases=self.use_biases,
            init_diag=self.init_diag,
            ti=self.ti
        )

    def call(self, input):
        alist = []
        n = self.n

        # Contracting both MPSs on each site to get local transfer matrices
        # Also adding the bias term at each site
        for i in range(n):
            mps, bias = self.get_mps(i)
            # Here we assume that the MPS has a bond dimension 1
            A = tf.einsum("...ijk,ljm->...ilkm", input[i], mps)
            if self.use_biases:
                A += bias
            if i == self.iout:
                A = tf.einsum("bij,...kjlm->...bkilm", self.Aout, A)
            alist.append(A)

        # Contracting neighbouring pairs of matrices until only one remains
        for i in range(int(np.log2(n)) + 1):
            blist = []
            for i in range(len(alist) // 2):
                A1 = alist[2 * i]
                A2 = alist[2 * i + 1]
                if len(A1.shape) == 6:
                    blist.append(tf.einsum("abijkl,aklmn->abijmn", A1, A2))
                elif len(A2.shape) == 6:
                    blist.append(tf.einsum("aijkl,abklmn->abijmn", A1, A2))
                else:
                    blist.append(tf.einsum("aijkl,aklmn->aijmn", A1, A2))

            if len(alist) % 2 == 1:
                blist.append(alist[-1])
            alist = blist
            if len(alist) == 1:
                break

        # Returning the trace of the remaining matrix
        res = tf.einsum("abijij->ab", alist[0])
        if self.boundary == "c":
            res = res / self.D

        if self.normalize:
            res = tf.abs(res)
            res = res / tf.reduce_sum(res, axis=1, keepdims=True)

        return res


def measurement_kernels(
    n,
    D,
    nout,
    iout,
    d=2,
    boundary="c",
    eps=1e-9,
    use_biases=False,
    dtype=tf.float32,
    trainable=True,
    init_diag=1.0,
    ti=False
):

    if ti:
        Dl = 1
        Dr = 1
        biases = []
        if boundary == "c":
            Dl = D
            Dr = D

        ashape = [Dl, d, D]
        Al = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
            init_diag * tf.eye(Dl, D, batch_shape=[d]), perm=[1, 0, 2])
        Al = tf.Variable(Al, trainable=trainable, dtype=dtype, name=f"Al")

        ashape = [D, d, D]
        A = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
            init_diag * tf.eye(D, D, batch_shape=[d]), perm=[1, 0, 2])
        A = tf.Variable(A, trainable=trainable, dtype=dtype, name=f"A")

        ashape = [D, d, Dr]
        Ar = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
            init_diag * tf.eye(D, Dr, batch_shape=[d]), perm=[1, 0, 2])
        Ar = tf.Variable(Ar, trainable=trainable, dtype=dtype, name=f"Ar")

        mps = [Al, A, Ar]

        if use_biases:
            bshape = [1, 1, Dl, 1, D]
            Bl = np.random.normal(loc=0, scale=eps, size=bshape)
            Bl[0, 0, :, 0, :] += np.eye(Dl, D)
            Bl = tf.Variable(Bl, dtype=dtype, name=f"Bl")

            bshape = [1, 1, D, 1, D]
            B = np.random.normal(loc=0, scale=eps, size=bshape)
            B[0, 0, :, 0, :] += np.eye(D, D)
            B = tf.Variable(B, dtype=dtype, name=f"B")

            bshape = [1, 1, D, 1, Dr]
            Br = np.random.normal(loc=0, scale=eps, size=bshape)
            Br[0, 0, :, 0, :] += np.eye(D, Dr)
            Br = tf.Variable(Br, dtype=dtype, name=f"Br")

            biases = [Bl, B, Br]

        Aout = tf.Variable(
            tf.eye(D, D, batch_shape=[nout]) +
            tf.random.normal([nout, D, D], mean=0, stddev=eps),
            trainable=trainable,
            dtype=dtype,
            name=f"Aout"
        )

        return mps, biases, Aout
    else:
        if boundary == "c":
            dims = (n + 1) * [D]
        elif boundary == "o":
            dims = [1] + [bond_dimension(D, d, n, i) for i in range(n)]
        else:
            logging.warning(
                "Unknown boundary condition. Using the periodic boundary condition")
            dims = (n + 1) * [D]
        mps = []
        biases = None
        Aout = None
        if use_biases:
            biases = []
        for i in range(n):
            dl = dims[i]
            dr = dims[i + 1]
            ashape = [dl, d, dr]
            A = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
                init_diag * tf.eye(dl, dr, batch_shape=[d]), perm=[1, 0, 2])
            mps.append(tf.Variable(A, trainable=trainable,
                                   dtype=dtype, name=f"A{i}"))
            if i == iout:
                Aout = tf.Variable(
                    tf.eye(dl, dl, batch_shape=[nout]) +
                    tf.random.normal([nout, dl, dl], mean=0, stddev=eps),
                    trainable=trainable,
                    dtype=dtype,
                    name=f"Aout{i}"
                )
            if use_biases:
                bshape = [1, 1, dl, 1, dr]
                B = np.random.normal(loc=0, scale=eps, size=bshape)
                B[0, 0, :, 0, :] += np.eye(dl, dr)
                biases.append(tf.Variable(B, dtype=dtype), name=f"B{i}")

        return mps, biases, Aout
