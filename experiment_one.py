import cirq

from cirq.aqt import aqt_device
from cirq.aqt import AQTSimulator

import optimizers.measurements_last as ml
import optimizers.replace_opposite as ro

def gen_s_dist_circuit(cirq_circuit, qubits):
    # fig 32 page 38 https://arxiv.org/pdf/1208.0928.pdf
    # The order of the qubits from that figure is a bit changed in this script
    # therein the lowest qubit is 0, because the Steane qubits are numbered from 1 to 7
    # thus, in order to keep things easy, zero is herein the output, but Cirq will draw it
    # due to its index as the highest wire and not the lowest wire

    # qubits are automatically initialized to 0
    cirq_circuit.append(cirq.ops.H(qubits[0]))
    # Steane code plus states
    cirq_circuit.append(cirq.ops.H(qubits[1]))
    cirq_circuit.append(cirq.ops.H(qubits[2]))
    cirq_circuit.append(cirq.ops.H(qubits[3]))

    # Bell pair
    cirq_circuit.append(cirq.ops.CNOT(qubits[0], qubits[7]))

    # first cnot
    cirq_circuit.append(cirq.ops.CNOT(qubits[7], qubits[4]))
    cirq_circuit.append(cirq.ops.CNOT(qubits[7], qubits[5]))

    # second cnot
    cirq_circuit.append(cirq.ops.CNOT(qubits[3], qubits[4]))
    cirq_circuit.append(cirq.ops.CNOT(qubits[3], qubits[5]))
    cirq_circuit.append(cirq.ops.CNOT(qubits[3], qubits[6]))

    # third cnot
    cirq_circuit.append(cirq.ops.CNOT(qubits[2], qubits[5]))
    cirq_circuit.append(cirq.ops.CNOT(qubits[2], qubits[6]))
    cirq_circuit.append(cirq.ops.CNOT(qubits[2], qubits[7]))

    # fourth cnot
    cirq_circuit.append(cirq.ops.CNOT(qubits[1], qubits[4]))
    cirq_circuit.append(cirq.ops.CNOT(qubits[1], qubits[6]))
    cirq_circuit.append(cirq.ops.CNOT(qubits[1], qubits[7]))

    # # round of S gates on the qubits numbered 1...7
    # for i in range(1, 8):
    #     cirq_circuit.append(cirq.ops.S(qubits[i]))

    # round of X measurements on the qubits numbered 1...7
    for i in range(1, 8):
        cirq_circuit.append(cirq.ops.H(qubits[i]))
        cirq_circuit.append(cirq.ops.MeasurementGate(1, key=str(i)).on(qubits[i]))

    return


def compute_parity(config, results) -> []:
    """
    Compute parity of measurement results
    config is a set of chip qubit indices to be considered in the parity calculation
    results is the map returned by get_counts() API call

    Returns:
        a   list where each measurement result has an associated parity accoding to config
    Example:
        config is [1, 3] and result is "1010" (see Discussion at getResultIndex()),
    thus parity is 1 - 2 * ( (1 + 1) %2) = 1 and method will return {"1010" : 1}
    """
    par = []
    for repetition in range(results.repetitions):
        parity = 0

        # which order are the results returned?
        for index in config:
            parity += int(results.measurements[str(index)][repetition])

        # 0 -> 1 (even) and 1 -> -1 (odd)
        par.append(1 - 2 * (parity % 2))

    return par


def generate_stats(results, parities):
    """
    Generate the statistics for a result map after receiving parity maps (see Discussion in computeParity())
    Method is not general, and is tailored for Y state distillation
    Receives three parity maps corresponding to the three stabilisers measured,
    and another total parity map of the results
    Returns: a map of the type {"1010" : [5, 1]}, where "1010" is measurement result successfully corrected,
    5 is its frequency as returned by get_counts(), and 1 and -1 indicate if the output Y state requires a Z correction
    """

    stats = {}
    # for each result returned from the simulation
    for repetition in range(results.repetitions):

        totalpar = 1
        str_par = ""
        for parity in parities.keys():
            if str(parity)[0] != "_":
                # If this is not a hidden key, such as _byprod
                par_bit = parities[parity][repetition]
                totalpar *= par_bit
                str_par += str(par_bit)

        str_meas = ""
        for k in results.measurements.keys():
            str_meas += str(results.measurements[k][repetition][0])

        print(str(repetition) + ". " + str_meas + " p:" + str_par + " total:" + str(totalpar))

        # these are the error-corrected states
        # store them in a map
        if totalpar == 1:
            # does the state require a pauli correction?
            entry = [0, parities["_byprod"][repetition]]
            if str_meas in stats.keys():
                entry = stats[str_meas]
            entry[0] += 1

            stats[str_meas] = entry
        else:
            print("not even parity")

    total = 0
    for result in stats:
        total += stats[result][0]

    print("Total states where parity of parities is even " + str(total))

    return stats


def main():
    print("Hello World!")

    # The current experiment uses only 8 qubits
    num_qubits = 8

    # Initialise the device and the qubits
    aqt_comp, aqt_qubits = aqt_device.get_aqt_device(num_qubits)

    # Create the circuit
    cirq_circuit = cirq.Circuit(device = aqt_comp)

    # Should send a parameter/dict for the gate that I wish to transversally apply on the nodes
    gen_s_dist_circuit(cirq_circuit, qubits=aqt_qubits)

    # Et voila! after the circuit has been automatically decomposed
    # because the IonDevice (and devices in general) call decompose_operation and decompose_circuit
    print(cirq_circuit)

    # is there an optimiser that cancels neighbouring gates of opposite angles?
    pointoptimizer = ro.ReplaceOppositeRotations()
    pointoptimizer.optimize_circuit(cirq_circuit)
    # print(cirq_circuit)

    # move measurements to the last moment
    measurementsopt = ml.MoveMeasurementsLastPass()
    measurementsopt.optimize_circuit(cirq_circuit)
    # print(cirq_circuit)


    # Instantiate a simulator
    # no_noise=True forces the underlying simulator to be a DensityMatrix without noise
    # Useful for checking correctness of circuit
    no_noise = True
    # Take the AQTSimulator
    aqt_sim = AQTSimulator(num_qubits=num_qubits, circuit = cirq_circuit, simulate_ideal=no_noise)
    # internally this runs a cirq.study
    results = aqt_sim.simulate_samples(repetitions=100)
    print(results)

    # perform analysis of measurement results
    print(" == Parities to check on chip qubits")

    multi_body_meas = {}
    multi_body_meas["plaq1"] = [3, 4, 5, 6]
    multi_body_meas["plaq2"] = [2, 5, 6, 7]
    multi_body_meas["plaq3"] = [1, 4, 6, 7]
    # this indicates if the code word has a flipped sign
    multi_body_meas["_byprod"] = list(range(1, 8))
    print("Multi bodty measurements are")
    print(multi_body_meas)

    plaq_parity = {}
    for plaq in multi_body_meas.keys():
        plaq_parity[plaq] = compute_parity(multi_body_meas[plaq], results)

    stats = generate_stats(results, plaq_parity)
    print("Statistics")
    print(stats)



if __name__ == "__main__":
    main()