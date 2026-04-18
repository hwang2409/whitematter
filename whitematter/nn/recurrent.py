"""Recurrent layers: LSTM, GRU."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module


class LSTM(Module):
    """Long Short-Term Memory network."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size
            bound = 1.0 / np.sqrt(hidden_size)
            # Input-hidden weights and bias (4 gates: i, f, g, o)
            setattr(self, f"weight_ih_{layer}", Tensor(
                np.random.uniform(-bound, bound, (4 * hidden_size, in_sz)).astype(np.float32),
                requires_grad=True,
            ))
            setattr(self, f"weight_hh_{layer}", Tensor(
                np.random.uniform(-bound, bound, (4 * hidden_size, hidden_size)).astype(np.float32),
                requires_grad=True,
            ))
            setattr(self, f"bias_ih_{layer}", Tensor(
                np.zeros(4 * hidden_size, dtype=np.float32), requires_grad=True,
            ))
            b_hh = np.zeros(4 * hidden_size, dtype=np.float32)
            b_hh[hidden_size:2*hidden_size] = 1.0  # forget gate bias = 1
            setattr(self, f"bias_hh_{layer}", Tensor(b_hh, requires_grad=True))

    def forward(self, x: Tensor, hx=None):
        if self.batch_first:
            N, T, _ = x.shape
        else:
            T, N, _ = x.shape

        H = self.hidden_size

        if hx is None:
            h0 = np.zeros((self.num_layers, N, H), dtype=np.float32)
            c0 = np.zeros((self.num_layers, N, H), dtype=np.float32)
        else:
            h0, c0 = hx[0].data, hx[1].data

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -88, 88)))

        all_params = []
        current_input = x.data
        if not self.batch_first:
            current_input = current_input.transpose(1, 0, 2)  # -> (N, T, input_size)

        all_outputs = None
        final_h = np.zeros((self.num_layers, N, H), dtype=np.float32)
        final_c = np.zeros((self.num_layers, N, H), dtype=np.float32)

        layer_inputs = []
        layer_gates = []

        for layer in range(self.num_layers):
            w_ih = getattr(self, f"weight_ih_{layer}").data
            w_hh = getattr(self, f"weight_hh_{layer}").data
            b_ih = getattr(self, f"bias_ih_{layer}").data
            b_hh = getattr(self, f"bias_hh_{layer}").data

            h_t = h0[layer].copy()
            c_t = c0[layer].copy()
            outputs = np.zeros((N, T, H), dtype=np.float32)

            gates_all = []
            h_states = [h_t.copy()]
            c_states = [c_t.copy()]

            for t in range(T):
                x_t = current_input[:, t, :]  # (N, in_sz)
                gates = x_t @ w_ih.T + b_ih + h_t @ w_hh.T + b_hh  # (N, 4H)

                i_gate = sigmoid(gates[:, :H])
                f_gate = sigmoid(gates[:, H:2*H])
                g_gate = np.tanh(gates[:, 2*H:3*H])
                o_gate = sigmoid(gates[:, 3*H:])

                c_t = f_gate * c_t + i_gate * g_gate
                h_t = o_gate * np.tanh(c_t)

                outputs[:, t, :] = h_t
                gates_all.append((i_gate, f_gate, g_gate, o_gate, gates))
                h_states.append(h_t.copy())
                c_states.append(c_t.copy())

            layer_inputs.append(current_input)
            layer_gates.append((gates_all, h_states, c_states))
            current_input = outputs
            final_h[layer] = h_t
            final_c[layer] = c_t

        if self.batch_first:
            all_outputs = current_input  # (N, T, H)
        else:
            all_outputs = current_input.transpose(1, 0, 2)

        params = []
        for layer in range(self.num_layers):
            params.extend([
                getattr(self, f"weight_ih_{layer}"),
                getattr(self, f"weight_hh_{layer}"),
                getattr(self, f"bias_ih_{layer}"),
                getattr(self, f"bias_hh_{layer}"),
            ])
        result = x._make_result(all_outputs, [x] + params)

        if result.requires_grad:
            lstm_self = self

            def _backward(grad):
                if not self.batch_first:
                    grad = grad.transpose(1, 0, 2)  # -> (N, T, H)

                d_input = grad  # gradient flowing back through layers

                for layer in reversed(range(lstm_self.num_layers)):
                    w_ih = getattr(lstm_self, f"weight_ih_{layer}")
                    w_hh = getattr(lstm_self, f"weight_hh_{layer}")
                    b_ih = getattr(lstm_self, f"bias_ih_{layer}")
                    b_hh = getattr(lstm_self, f"bias_hh_{layer}")

                    gates_all, h_states, c_states = layer_gates[layer]
                    inp = layer_inputs[layer]

                    dh_next = np.zeros((N, H), dtype=np.float32)
                    dc_next = np.zeros((N, H), dtype=np.float32)
                    d_inp = np.zeros_like(inp)

                    dw_ih = np.zeros_like(w_ih.data)
                    dw_hh = np.zeros_like(w_hh.data)
                    db_ih = np.zeros(4 * H, dtype=np.float32)
                    db_hh = np.zeros(4 * H, dtype=np.float32)

                    for t in reversed(range(T)):
                        i_g, f_g, g_g, o_g, raw_gates = gates_all[t]
                        h_prev = h_states[t]
                        c_prev = c_states[t]
                        c_cur = c_states[t + 1]

                        dh = d_input[:, t, :] + dh_next
                        tanh_c = np.tanh(c_cur)
                        do = dh * tanh_c
                        dc = dh * o_g * (1 - tanh_c ** 2) + dc_next
                        di = dc * g_g
                        df = dc * c_prev
                        dg = dc * i_g
                        dc_next = dc * f_g

                        di_raw = di * i_g * (1 - i_g)
                        df_raw = df * f_g * (1 - f_g)
                        dg_raw = dg * (1 - g_g ** 2)
                        do_raw = do * o_g * (1 - o_g)

                        d_gates = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=1)  # (N, 4H)

                        db_ih += d_gates.sum(axis=0)
                        db_hh += d_gates.sum(axis=0)
                        dw_ih += d_gates.T @ inp[:, t, :]
                        dw_hh += d_gates.T @ h_prev

                        d_inp[:, t, :] = d_gates @ w_ih.data
                        dh_next = d_gates @ w_hh.data

                    if w_ih.requires_grad:
                        w_ih.grad = _acc(w_ih.grad, dw_ih)
                    if w_hh.requires_grad:
                        w_hh.grad = _acc(w_hh.grad, dw_hh)
                    if b_ih.requires_grad:
                        b_ih.grad = _acc(b_ih.grad, db_ih)
                    if b_hh.requires_grad:
                        b_hh.grad = _acc(b_hh.grad, db_hh)

                    d_input = d_inp

                if x.requires_grad:
                    if not self.batch_first:
                        d_input = d_input.transpose(1, 0, 2)
                    x.grad = _acc(x.grad, d_input)

            result._grad_fn = _backward

        h_out = Tensor(final_h)
        c_out = Tensor(final_c)
        return result, (h_out, c_out)

    def __repr__(self):
        return f"LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})"


class GRU(Module):
    """Gated Recurrent Unit network."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size
            bound = 1.0 / np.sqrt(hidden_size)
            setattr(self, f"weight_ih_{layer}", Tensor(
                np.random.uniform(-bound, bound, (3 * hidden_size, in_sz)).astype(np.float32),
                requires_grad=True,
            ))
            setattr(self, f"weight_hh_{layer}", Tensor(
                np.random.uniform(-bound, bound, (3 * hidden_size, hidden_size)).astype(np.float32),
                requires_grad=True,
            ))
            setattr(self, f"bias_ih_{layer}", Tensor(
                np.zeros(3 * hidden_size, dtype=np.float32), requires_grad=True,
            ))
            setattr(self, f"bias_hh_{layer}", Tensor(
                np.zeros(3 * hidden_size, dtype=np.float32), requires_grad=True,
            ))

    def forward(self, x: Tensor, h0_tensor=None):
        if self.batch_first:
            N, T, _ = x.shape
        else:
            T, N, _ = x.shape

        H = self.hidden_size

        if h0_tensor is None:
            h0 = np.zeros((self.num_layers, N, H), dtype=np.float32)
        else:
            h0 = h0_tensor.data

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -88, 88)))

        current_input = x.data
        if not self.batch_first:
            current_input = current_input.transpose(1, 0, 2)

        layer_inputs = []
        layer_gates = []
        final_h = np.zeros((self.num_layers, N, H), dtype=np.float32)

        for layer in range(self.num_layers):
            w_ih = getattr(self, f"weight_ih_{layer}").data
            w_hh = getattr(self, f"weight_hh_{layer}").data
            b_ih = getattr(self, f"bias_ih_{layer}").data
            b_hh = getattr(self, f"bias_hh_{layer}").data

            h_t = h0[layer].copy()
            outputs = np.zeros((N, T, H), dtype=np.float32)
            gates_all = []
            h_states = [h_t.copy()]

            for t in range(T):
                x_t = current_input[:, t, :]
                gi = x_t @ w_ih.T + b_ih  # (N, 3H)
                gh = h_t @ w_hh.T + b_hh  # (N, 3H)

                r_gate = sigmoid(gi[:, :H] + gh[:, :H])
                z_gate = sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
                n_gate = np.tanh(gi[:, 2*H:] + r_gate * gh[:, 2*H:])

                h_t = (1 - z_gate) * n_gate + z_gate * h_t
                outputs[:, t, :] = h_t

                gates_all.append((r_gate, z_gate, n_gate, gi, gh))
                h_states.append(h_t.copy())

            layer_inputs.append(current_input)
            layer_gates.append((gates_all, h_states))
            current_input = outputs
            final_h[layer] = h_t

        if self.batch_first:
            all_outputs = current_input
        else:
            all_outputs = current_input.transpose(1, 0, 2)

        params = []
        for layer in range(self.num_layers):
            params.extend([
                getattr(self, f"weight_ih_{layer}"),
                getattr(self, f"weight_hh_{layer}"),
                getattr(self, f"bias_ih_{layer}"),
                getattr(self, f"bias_hh_{layer}"),
            ])
        result = x._make_result(all_outputs, [x] + params)

        if result.requires_grad:
            gru_self = self

            def _backward(grad):
                if not self.batch_first:
                    grad = grad.transpose(1, 0, 2)

                d_input = grad

                for layer in reversed(range(gru_self.num_layers)):
                    w_ih = getattr(gru_self, f"weight_ih_{layer}")
                    w_hh = getattr(gru_self, f"weight_hh_{layer}")
                    b_ih = getattr(gru_self, f"bias_ih_{layer}")
                    b_hh = getattr(gru_self, f"bias_hh_{layer}")

                    gates_all, h_states = layer_gates[layer]
                    inp = layer_inputs[layer]

                    dh_next = np.zeros((N, H), dtype=np.float32)
                    d_inp = np.zeros_like(inp)

                    dw_ih = np.zeros_like(w_ih.data)
                    dw_hh = np.zeros_like(w_hh.data)
                    db_ih = np.zeros(3 * H, dtype=np.float32)
                    db_hh = np.zeros(3 * H, dtype=np.float32)

                    for t in reversed(range(T)):
                        r_g, z_g, n_g, gi, gh = gates_all[t]
                        h_prev = h_states[t]

                        dh = d_input[:, t, :] + dh_next

                        # dh/dz = h_prev - n
                        dz = dh * (h_prev - n_g)
                        # dh/dn = (1 - z)
                        dn = dh * (1 - z_g)
                        # dh/dh_prev through z
                        dh_next = dh * z_g

                        # Through tanh
                        dn_raw = dn * (1 - n_g ** 2)
                        # Through sigmoid for z
                        dz_raw = dz * z_g * (1 - z_g)

                        # n = tanh(gi[2H:] + r * gh[2H:])
                        dr_gh = dn_raw * gh[:, 2*H:]  # gradient of r * gh[2H:]
                        dr = dr_gh  # since gh[2H:] is the coefficient, dr = dn_raw * gh[2H:]
                        dr_raw = dr * r_g * (1 - r_g)

                        # Gradients for gi and gh
                        d_gi = np.zeros_like(gi)
                        d_gh = np.zeros_like(gh)

                        d_gi[:, :H] = dr_raw
                        d_gi[:, H:2*H] = dz_raw
                        d_gi[:, 2*H:] = dn_raw

                        d_gh[:, :H] = dr_raw
                        d_gh[:, H:2*H] = dz_raw
                        d_gh[:, 2*H:] = dn_raw * r_g

                        db_ih += d_gi.sum(axis=0)
                        db_hh += d_gh.sum(axis=0)
                        dw_ih += d_gi.T @ inp[:, t, :]
                        dw_hh += d_gh.T @ h_prev

                        d_inp[:, t, :] = d_gi @ w_ih.data
                        dh_next += d_gh @ w_hh.data

                    if w_ih.requires_grad:
                        w_ih.grad = _acc(w_ih.grad, dw_ih)
                    if w_hh.requires_grad:
                        w_hh.grad = _acc(w_hh.grad, dw_hh)
                    if b_ih.requires_grad:
                        b_ih.grad = _acc(b_ih.grad, db_ih)
                    if b_hh.requires_grad:
                        b_hh.grad = _acc(b_hh.grad, db_hh)

                    d_input = d_inp

                if x.requires_grad:
                    if not self.batch_first:
                        d_input = d_input.transpose(1, 0, 2)
                    x.grad = _acc(x.grad, d_input)

            result._grad_fn = _backward

        return result, Tensor(final_h)

    def __repr__(self):
        return f"GRU(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})"
