from os import getenv

import cirq

import numpy as np

from cirq.aqt import AQTSampler, AQTSimulator


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
        cirq_circuit.append(cirq.measure(qubits[i]))

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
            parb = 0
            # if is_cloud_sim:
            #     parb += int(
            #         results.measurements[measurement_key][repetition][index])
            # else:
            parb += int(
                results.measurements[str(index)][repetition])

            parity += parb

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

        total_bit_all_plaques = 1
        str_plaque_par = ""
        for plaq_parity in parities.keys():
            if str(plaq_parity)[0] != "_":
                # If this is not a hidden key, such as _byprod
                par_bit = parities[plaq_parity][repetition]
                total_bit_all_plaques *= par_bit
                str_plaque_par += str(par_bit)

        qubits_meas_res = ""
        for k in results.measurements.keys():
            # I am not sure why the result is store in a 1D array
            qubits_meas_res += str(results.measurements[k][repetition][0])

        print(str(repetition) + ". " + qubits_meas_res
              + " plaq_par:" + str_plaque_par
              + " total:" + str(total_bit_all_plaques))

        # these are the error-corrected states
        # store them in a map
        if total_bit_all_plaques == 1:
            # does the state require a pauli correction?
            # create an entry
            entry = [0, parities["_byprod"][repetition]]
            # if the entry exists retrieve it
            if qubits_meas_res in stats.keys():
                entry = stats[qubits_meas_res]
            # update the count
            entry[0] += 1

            # this measurement pattern was observed according
            # to the data stored in the entry
            stats[qubits_meas_res] = entry
        else:
            # this did not go well
            print("not even parity")

    """
    What is the total number of times the even total parity was observed?
    """
    total = 0
    for result in stats:
        total += stats[result][0]

    print("Total states where parity of plaquetes is even " + str(total))

    return stats


def reformat_results(results):
    """
        AQTSampler stores repetition-qubit instead of qubit-repetition
        This method inverts rows with columns to be compatible with Cirq
    :param results:
    :return: inverted dictionary of measurement results
    """
    # The key by which the results are stored
    measurements = results.measurements["m"]

    # The measurement keys will be the index of the qubits
    nr_qubits = len(measurements[0])

    new_dict = {}
    for i in range(nr_qubits):
        entry = np.ndarray(shape=(results.repetitions, 1), dtype=np.int)
        for j in range(results.repetitions):
            entry[j] = int(measurements[j][i])

        new_dict[str(i)] = entry

    # Access private/hidden, because measurements is property
    results._measurements = new_dict

    return results

def main():
    print("Hello World!")

    # The current experiment uses only 8 qubits
    num_qubits = 8

    # Initialise the device and the qubits
    aqt_comp, aqt_qubits = cirq.aqt.aqt_device.get_aqt_device(num_qubits)

    # Create the circuit
    circuit = cirq.Circuit(device = aqt_comp)

    # Should send a parameter/dict for the gate that I wish to transversally apply on the nodes
    gen_s_dist_circuit(circuit, qubits=aqt_qubits)

    # Et voila! after the circuit has been automatically decomposed
    # because the IonDevice (and devices in general) call decompose_operation and decompose_circuit
    print(circuit)

    # is there an optimiser that cancels neighbouring gates of opposite angles?
    pointoptimizer = ro.ReplaceOppositeRotations()
    pointoptimizer.optimize_circuit(circuit)
    # print(cirq_circuit)

    # move measurements to the last moment
    measurementsopt = ml.MoveMeasurementsLastPass()
    measurementsopt.optimize_circuit(circuit)
    # print(cirq_circuit)


    # # Instantiate a simulator
    # # no_noise=True forces the underlying simulator to be a DensityMatrix without noise
    # # Useful for checking correctness of circuit
    # no_noise = True
    # # Take the AQTSimulator
    # aqt_sim = cirq.aqt.AQTSimulator(num_qubits=num_qubits, circuit = circuit, simulate_ideal=no_noise)
    # # internally this runs a cirq.study
    # results = aqt_sim.simulate_samples(repetitions=100)
    # print(results)
    #
    aqt_sampler = cirq.aqt.AQTSampler(
        remote_host = "https://gateway.aqt.eu/marmot/sim/noise-model-1",
        access_token = str(getenv("AQT_TOKEN"))
    )

    results = aqt_sampler.run(circuit, repetitions=100)
    results = reformat_results(results)
    print(results)

    # perform analysis of measurement results
    print(" == Parities to check on chip qubits")

    multi_body_meas = {}
    multi_body_meas["plaq1"] = [3, 4, 5, 6]
    multi_body_meas["plaq2"] = [2, 5, 6, 7]
    multi_body_meas["plaq3"] = [1, 4, 6, 7]
    # this indicates if the code word has a flipped sign
    multi_body_meas["_byprod"] = list(range(1, 8))
    print("Multi body measurements are")
    print(multi_body_meas)

    plaq_parity = {}
    for plaq in multi_body_meas.keys():
        plaq_parity[plaq] = compute_parity(multi_body_meas[plaq],
                                           results)

    stats = generate_stats(results, plaq_parity)
    print("Statistics")
    print(stats)


if __name__ == "__main__":
    main()