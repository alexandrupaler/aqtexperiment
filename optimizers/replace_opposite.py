import cirq

class ReplaceOppositeRotations(cirq.PointOptimizer):
    """
    Replaces  opposite rotations with identity.
    """
    def optimization_at(self, circuit, index, op):

        if not self.is_single_qubit_rotation(op):
            return None

        n_idx = circuit.next_moment_operating_on(op.qubits, index + 1)
        if n_idx is None:
            return None

        next_op = circuit.operation_at(op.qubits[0], n_idx)

        same_type = False
        if isinstance(op.gate, cirq.ops.XPowGate) and isinstance(next_op.gate, cirq.ops.XPowGate):
            same_type = True
        if isinstance(op.gate, cirq.ops.ZPowGate) and isinstance(next_op.gate, cirq.ops.ZPowGate):
            same_type = True
        if isinstance(op.gate, cirq.ops.YPowGate) and isinstance(next_op.gate, cirq.ops.YPowGate):
            same_type = True

        if not same_type:
            return None

        if op.gate.global_shift != next_op.gate.global_shift:
            return None

        if op.gate.exponent != -next_op.gate.exponent:
            return None


        return cirq.PointOptimizationSummary(clear_span=n_idx - index + 1,
                                        clear_qubits=op.qubits,
                                        new_operations=[])# Two opposite rotations are erased

    def is_single_qubit_rotation(self, op):
        return isinstance(op.gate, (cirq.ops.XPowGate, cirq.ops.YPowGate, cirq.ops.ZPowGate))