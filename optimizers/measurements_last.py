import cirq

class MoveMeasurementsLastPass():
    """
    An optimization pass that moves measurements to the last moment
    """

    def __init__(self) -> None:
        return

    def __call__(self, circuit: cirq.Circuit):
        self.optimize_circuit(circuit)

    def optimize_circuit(self, circuit: cirq.Circuit) -> None:
        # list of tuples [int, ops.Operation]
        deletions = []
        for moment_index, moment in enumerate(circuit):
            for op in moment.operations:
                if (op is not None) and isinstance (op.gate, cirq.ops.MeasurementGate):
                    deletions.append((moment_index, op))
        circuit.batch_remove(deletions)

        opss = []
        for oldm, op in deletions:
            opss.append(op)
        last_moment = cirq.Moment(opss)
        circuit.append(last_moment)

