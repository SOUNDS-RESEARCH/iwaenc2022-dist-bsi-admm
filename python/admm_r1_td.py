import numpy as np


class NodeProcessor:
    def __init__(self, id, filter_len, rho) -> None:
        # identifier for node
        self.id = id
        # iterator
        self.k = 0
        # local solver type
        self.type = type
        # definition of signal connections
        self.group = None
        # number of FIR filter taps
        self.L = filter_len
        # step size / penalty parameter
        self.rho = rho
        self.mu = 0.5
        self.eta = 0.95
        self.zeta = 0.95
        # number of "channels"
        self.N = None
        # signal block
        self.block = None
        # local primal
        self.x = None
        # local dual
        self.y = None
        # local dual est
        self.y_hat = None
        # local dual est old
        self.z_l = None
        # estimated
        self.H_i = None
        # buffer

        # self.setBufferSize(2 * self.L)
        pass

    def setLocalSize(self, N) -> None:
        self.N = N
        # global indices
        self.global_indices = None
        self.reset()
        pass

    def setBufferSize(self, buffer_size) -> None:
        self.buffer_size = buffer_size
        self.buffer = np.zeros((self.L, self.N, self.buffer_size))
        # self.reset()
        pass

    def reset(self) -> None:
        self.k = 0
        # signal block
        self.block = np.zeros(shape=(self.L, self.N))
        # local primal
        # self.x = np.zeros(shape=(self.L * self.N, 1))
        self.x = np.random.normal(size=(self.L * self.N, 1))
        # self.x[0 :: self.L] = 1
        # local dual
        self.y = np.zeros(shape=(self.L * self.N, 1))
        # local dual est
        # self.z_l = np.zeros(shape=(self.L*self.N, 1))
        # self.z_l[0::self.L] = 1
        # self.z_l = np.ones(shape=(self.L*self.N, 1))
        self.z_l = np.random.normal(size=(self.L * self.N, 1))
        # estimated cross-correlatino difference matrix
        self.R = np.zeros((self.L * self.N, self.L * self.N))
        self.R1 = np.zeros((self.L * self.N, self.L * self.N))
        self.H_i = np.eye(self.L * self.N) / self.rho
        self.H_i1 = np.eye(self.L * self.N) / self.rho
        self.first = True

        self.buffer_size = self.L
        self.buffer = np.zeros((self.L, self.N, self.buffer_size))
        self.buffer_index = 0

        self.R_diff = []
        self.H_diff = []
        self.gradient_h = []
        self.y_h = []
        self.J_hist = []

        pass

    def setRho(self, rho, mu, eta, zeta, scaling) -> None:
        self.rho = rho
        self.H_i = np.eye(self.L * self.N) / self.rho
        self.mu = mu
        self.eta = eta
        self.zeta = zeta
        self.scaling = scaling
        pass

    def receiveSignal(self, signal) -> None:
        self.block = signal[::-1] * self.scaling
        pass

    def step(self) -> None:
        self.solveLocalLS()
        pass

    def rank1Update(self) -> None:
        x_b = self.buffer[:, :, self.buffer_index]
        for m in range(self.N):
            for n in range(self.N):
                if m != n:
                    # correction step diagonal
                    self.R[
                        (n) * self.L : (n + 1) * self.L, (n) * self.L : (n + 1) * self.L
                    ] -= np.outer(x_b[:, m], x_b[:, m])

                    Vu = self.H_i[:, (n) * self.L : (n + 1) * self.L] @ x_b[:, m, None]
                    vTV = (
                        -x_b[:, m, None].T
                        @ self.H_i[(n * self.L) : (n + 1) * self.L, :]
                    )
                    self.H_i -= (
                        Vu
                        @ vTV
                        / (1 + vTV[:, n * self.L : (n + 1) * self.L] @ x_b[:, m, None])
                    )

                    # correction step off-diagonal
                    self.R[
                        (m) * self.L : (m + 1) * self.L, (n) * self.L : (n + 1) * self.L
                    ] += np.outer(x_b[:, n], x_b[:, m])

                    Vu = self.H_i[:, n * self.L : (n + 1) * self.L] @ x_b[:, m, None]
                    vTV = (
                        x_b[:, n, None].T @ self.H_i[(m * self.L) : (m + 1) * self.L, :]
                    )
                    self.H_i -= (
                        Vu
                        @ vTV
                        / (1 + vTV[:, n * self.L : (n + 1) * self.L] @ x_b[:, m, None])
                    )

                    # update step diagonal
                    self.R[
                        (n) * self.L : (n + 1) * self.L, (n) * self.L : (n + 1) * self.L
                    ] += np.outer(self.block[:, m], self.block[:, m])

                    Vu = (
                        self.H_i[:, n * self.L : (n + 1) * self.L]
                        @ self.block[:, m, None]
                    )
                    vTV = (
                        self.block[:, m, None].T
                        @ self.H_i[n * self.L : (n + 1) * self.L, :]
                    )
                    self.H_i -= (
                        Vu
                        @ vTV
                        / (
                            1
                            + vTV[:, n * self.L : (n + 1) * self.L]
                            @ self.block[:, m, None]
                        )
                    )

                    # update step off-diagonal
                    self.R[
                        (m) * self.L : (m + 1) * self.L, (n) * self.L : (n + 1) * self.L
                    ] -= np.outer(self.block[:, n], self.block[:, m])

                    Vu = (
                        self.H_i[:, m * self.L : (m + 1) * self.L]
                        @ self.block[:, n, None]
                    )
                    vTV = (
                        -self.block[:, m, None].T
                        @ self.H_i[(n * self.L) : (n + 1) * self.L, :]
                    )
                    self.H_i -= (
                        Vu
                        @ vTV
                        / (
                            1
                            + vTV[:, m * self.L : (m + 1) * self.L]
                            @ self.block[:, n, None]
                        )
                    )

        self.buffer[:, :, self.buffer_index] = self.block.copy()
        self.buffer_index = self.buffer_index + 1
        if self.buffer_index >= self.buffer_size:
            self.buffer_index -= self.buffer_size
        pass

    def solveLocalLS(self) -> None:
        self.rank1Update()

        self.R_ = self.R if self.first else self.eta * self.R_ + (1 - self.eta) * self.R
        self.H_i_ = (
            self.H_i
            if self.first
            else self.zeta * self.H_i_ + (1 - self.zeta) * self.H_i
        )
        self.first = False

        self.R_diff.append(np.linalg.norm(self.R_))
        self.H_diff.append(np.linalg.norm(self.H_i_))

        J1 = self.x.T @ self.R_ @ self.x
        J2 = self.y.T @ (self.x - self.z_l)
        J3 = self.rho / 2 * np.linalg.norm(self.x - self.z_l) ** 2
        self.J_hist.append([J1.squeeze(), J2.squeeze(), J3.squeeze()])

        y = self.R_ @ self.x + self.y + self.rho * (self.x - self.z_l)
        self.gradient_h.append(np.linalg.norm(y))
        self.x = self.x - self.mu * self.H_i_ @ y
        pass

    def dualUpdate(self) -> None:
        self.y = self.y + self.rho * (self.x - self.z_l)
        self.y_h.append(np.linalg.norm(self.y))
        pass


class Network:
    def __init__(self, L) -> None:
        # number of FIR filter taps
        self.L = L
        # number of nodes in network
        self.N = 0
        # central processor for gloabl update
        self.central_processor = None
        # node objecs
        self.nodes = {}
        # connections between nodes
        self.connections = {}
        # node index
        self.node_index = {}
        # global primal variable
        self.z = None
        # network matrix
        self.A = None
        self.A_ = None
        # global update weights
        self.g = None
        pass

    def reset(self) -> None:
        # self.z = np.zeros(shape=(self.N*self.L, 1))
        for node_key, node in self.nodes.items():
            node.reset()
        self.zs = np.array([]).reshape(0, self.L * self.N)
        pass

    def addNode(self, id, rho) -> int:
        self.nodes[id] = NodeProcessor(id, self.L, rho)
        self.N = len(self.nodes)
        self.z = np.zeros(shape=(self.N * self.L, 1))
        self.generateNetworkData()
        pass

    def removeNode(self, id) -> None:
        del self.nodes[id]
        self.N = len(self.nodes)
        self.z = np.zeros(shape=(self.N * self.L, 1))
        self.generateNetworkData()
        pass

    def setConnection(self, node_id, connections) -> None:
        self.connections[node_id] = connections
        self.generateNetworkData()
        pass

    def generateNetworkData(self) -> None:
        self.A = np.zeros(shape=(self.N, self.N))
        self.A_ = np.diag(np.repeat(1, self.N))
        self.node_index = {}
        for i, node_key in enumerate(self.nodes):
            self.node_index[node_key] = i

        for i, node_key in enumerate(self.nodes):
            if node_key in self.connections:
                for connection in self.connections[node_key]:
                    j = self.node_index[connection]
                    self.A[i, j] = 1
                    self.A_[i, j] = 1

        for node_key in self.node_index:
            i = self.node_index[node_key]
            self.nodes[node_key].setLocalSize(np.sum(self.A_[:, i]))
            self.nodes[node_key].global_indices = (
                np.tile(np.arange(self.L), self.nodes[node_key].N)
                + np.repeat(np.where(self.A_[:, i]), self.L) * self.L
            )

        self.g = np.repeat(1 / np.sum(self.A_, axis=1), self.L).reshape(
            self.N * self.L, 1
        )
        pass

    def step(self, signal) -> None:
        self.broadcastSignals(signal)
        self.localPrimalUpdate()
        self.globalUpdate()
        self.broadcastGlobalVariable()
        self.localDualUpdate()
        self.zs = np.concatenate([self.zs, self.z.T])
        pass

    def broadcastSignals(self, signal) -> None:
        for node_key, node in self.nodes.items():
            i = self.node_index[node_key]
            local_signal = signal[:, self.A_[:, i] == 1]
            node.receiveSignal(local_signal)
        pass

    def localPrimalUpdate(self) -> None:
        for node_key, node in self.nodes.items():
            node.step()
        pass

    def globalUpdate(self) -> None:
        self.z = np.zeros(shape=(self.N * self.L, 1))
        x_avg = np.zeros(shape=(self.N * self.L, 1))
        y_avg = np.zeros(shape=(self.N * self.L, 1))
        for node_key, node in self.nodes.items():
            x_avg[node.global_indices] += node.x * self.g[node.global_indices]
            y_avg[node.global_indices] += node.y * self.g[node.global_indices]
        pp = node.rho * x_avg + y_avg  # WHAT DO WITH RHO IF NOT SAME FOR ALL
        self.z = pp / np.linalg.norm(pp)
        pass

    def broadcastGlobalVariable(self) -> None:
        for node_key, node in self.nodes.items():
            node.z_l = self.z[node.global_indices]
        pass

    def localDualUpdate(self) -> None:
        for node_key, node in self.nodes.items():
            node.dualUpdate()
        pass

    def setBufferSize(self, buffer_size) -> None:
        for node_key, node in self.nodes.items():
            node.setBufferSize(buffer_size)
        pass

    def setRho(self, rho, mu, eta, zeta, scaling) -> None:
        for node_key, node in self.nodes.items():
            node.setRho(rho, mu, eta, zeta, scaling)
        pass
