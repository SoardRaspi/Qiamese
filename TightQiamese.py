import jax
from jax import jit
import jax.numpy as jnp
import optax
from jax import vmap, jit, vjp
from jax.extend import backend  # New 2026 replacement for xla_bridge
from jax.dlpack import from_dlpack as d2j, to_dlpack as j2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack as f2d, to_dlpack as t2d

import pennylane as qml
import tensorcircuit as tc
import tensorcircuit.gates as tcg

import numpy as np
import matplotlib.pyplot as plt
import yaml
import easydict
import os
import sys
import logging

K = tc.set_backend("jax")



def pauli_matrix(p: str):
    paulis = {
        'I': jnp.eye(2, dtype=jnp.complex128),
        'X': jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128),
        'Y': jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128),
        'Z': jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128),
    }
    
    return paulis[p]

def Oki_block(tc_circ : tc.Circuit, data_qubits : int, switch : int, anc_qubits : int, 
              k : int, i : int, 
              obsv_string, obsv_coeff=None):
    """
    tc_circ : the tensorcircuit circuit object
    data_qubits : number of data qubits in the circuit
    switch : index of the switch wire, index starts from 0
    anc_qubits : number of ancilla qubits in the circuit
    k : type of the observable block -> {1, 2, 3}
    i : index of the block index or iteration, indexing starts from 1
    obsv_string : list of pauli operators as letters : I, X, Y, Z; first pauli is operated on qubit index 0 and so on...

    returns the final circuit after all the operations

    TODO: add support for custom observables as well using LCU or Block-Encoding is non-unitary observable required
    """
    assert i >= 1
    assert switch == data_qubits
    
    if len(obsv_string) < data_qubits:
        print("length of input observable is smaller than number of data wires")
        print("obsevale modified as: " + obsv_string + "" * (data_qubits - len(obsv_string)))
    elif len(obsv_string) > data_qubits:
        raise ValueError(f"Observable string of incorrect length given as parameter : {len(obsv_string)} > {data_qubits}")
    
    if k == 1:
        # type 1 of controlled observable
        assert i <= anc_qubits
        
        control_wire = switch + i
        
        for target_wire, pauli in enumerate(obsv_string):
            if pauli == 'I':
                continue
            elif pauli == 'Z':
                tc_circ.cz(control_wire, target_wire)
            
            elif pauli == 'X':
                tc_circ.cx(control_wire, target_wire)
            
            elif pauli == 'Y':
                tc_circ.cy(control_wire, target_wire)
            
            else:
                raise ValueError(f"Unknown Pauli '{pauli}' at position {idx}. Use I, X, Y, Z.")
    elif k == 2:
        # type 2 of controlled observable
        assert i <= (1 << anc_qubits)

        control_wires  = list(range(switch + 1, switch + anc_qubits + 1))
        control_values = [int(b) for b in format(i, f'0{anc_qubits}b')][::-1]
        
        for target_wire, pauli in enumerate(obsv_string):
            mc_gate = pauli_matrix('I')
            if pauli == 'I':
                continue
            elif pauli == 'Z':
                mc_gate = tcg.multicontrol_gate(pauli_matrix('Z'), ctrl=control_values)
            
            elif pauli == 'X':
                mc_gate = tcg.multicontrol_gate(pauli_matrix('X'), ctrl=control_values)
            
            elif pauli == 'Y':
                mc_gate = tcg.multicontrol_gate(pauli_matrix('Y'), ctrl=control_values)
            
            else:
                raise ValueError(f"Unknown Pauli '{pauli}' at position {idx}. Use I, X, Y, Z.")

            tc_circ.any(*control_wires, target_wire, unitary=mc_gate)
    elif k == 3:
        # type 3 of controlled observable

        control_wires  = list(range(switch, switch + anc_qubits + 1))
        control_values = [0] * (anc_qubits + 1)
        
        for target_wire, pauli in enumerate(obsv_string):
            mc_gate = pauli_matrix('I')
            if pauli == 'I':
                continue
            elif pauli == 'Z':
                mc_gate = tcg.multicontrol_gate(pauli_matrix('Z'), ctrl=control_values)
            
            elif pauli == 'X':
                mc_gate = tcg.multicontrol_gate(pauli_matrix('X'), ctrl=control_values)
            
            elif pauli == 'Y':
                mc_gate = tcg.multicontrol_gate(pauli_matrix('Y'), ctrl=control_values)
            
            else:
                raise ValueError(f"Unknown Pauli '{pauli}' at position {idx}. Use I, X, Y, Z.")

            tc_circ.any(*control_wires, target_wire, unitary=mc_gate)
    else:
        raise ValueError("Incorrect type of observable block given as parameter.")
    
    return tc_circ

def circuit_tc(inputs, vqc_weights, mixing_weights=None):
    # Setup indexing and constants
    switch = 10 # switch qubit index
    # ancilla = 11 # parity ancilla index
    n_qubits = 11
    
    # Initialize Circuit
    c = tc.Circuit(n_qubits)
    
    num_vqc_blocks = vqc_weights.shape[0]
    num_layers = vqc_weights.shape[1]
    
    c.h(switch)
    
    # # 1. Angular Embedding Logic
    # # PennyLane: qml.ctrl(AngleEmbedding, switch, control_values=...)
    # # TC: Explicit Controlled-RY for the drug and protein channels
    # # Channel A (Protein)
    # for i in range(switch):
    #     # Apply RY(inputs[i]) if switch is 0 (Drug-Protein Context)
    #     c.cry(switch, i, theta=inputs[i] * jnp.pi) 
    #     # Note: tc.cry is controlled-RY. To do control-on-0, 
    #     # we usually X the switch, cry, then X back.
    #     c.x(switch)
    #     c.cry(switch, i, theta=inputs[i+switch] * jnp.pi)
    #     c.x(switch)

    # 2. VQC Blocks (StronglyEntanglingLayers)
    for b in range(num_vqc_blocks):
        if b == 0:
            # 1. Angular Embedding Logic
            # PennyLane: qml.ctrl(AngleEmbedding, switch, control_values=...)
            # TC: Explicit Controlled-RY for the drug and protein channels
            # Channel A (Protein)
            for i in range(switch):
                # Apply RY(inputs[i]) if switch is 0 (Drug-Protein Context)
                c.cry(switch, i, theta=inputs[i] * jnp.pi) 
                # Note: tc.cry is controlled-RY. To do control-on-0, 
                # we usually X the switch, cry, then X back.
            for i in range(switch):
                c.x(switch)
                c.cry(switch, i, theta=inputs[i+switch] * jnp.pi)
                c.x(switch)
        else:
            for i in range(switch):
                # Apply RY(inputs[i]) if switch is 0 (Drug-Protein Context)
                c.cry(switch, i, theta=inputs[i+switch]) 
                # Note: tc.cry is controlled-RY. To do control-on-0, 
                # we usually X the switch, cry, then X back.
            for i in range(switch):
                c.x(switch)
                c.cry(switch, i, theta=inputs[i])
                c.x(switch)
        
        # # Optional: Apply mixing_weights Rot gate on switch if b > 0
        # if b > 0 and mixing_weights is not None:
        #     c.r(switch, theta=mixing_weights[b-1, 0], 
        #           alpha=mixing_weights[b-1, 1], 
        #           phi=mixing_weights[b-1, 2])

        # # 1. Initial Hadamard layer to create superposition
        # for i in range(switch):
        #     c.h(i)
        
        # for l in range(num_layers):
        #     # Single-qubit phases
        #     for i in range(switch):
        #         c.rz(i, theta=vqc_weights[b, l, i, 0])
        
        #     # All-to-all entanglement (Non-linear interactions)
        #     # This captures the "Cross-talk" between all features
        #     for i in range(switch):
        #         for j in range(i + 1, switch):
        #             # We reuse weight indices or use a specific mapping
        #             c.crz(i, j, theta=vqc_weights[b, l, i, 1] * vqc_weights[b, l, j, 2])
            
        #     # # 2. Diagonal Rotations (Z-phase)
        #     # # Using the first index of your existing weights [..., 0]
        #     # for i in range(switch):
        #     #     c.rz(i, theta=vqc_weights[b, l, i, 0])
        
        #     # # 3. Diagonal Entanglement (Controlled-Z rotations)
        #     # # This creates the "Polynomial" interactions
        #     # for i in range(switch):
        #     #     # Using the second index [..., 1] for entanglement phases
        #     #     # Pairs: (0,1), (1,2) ... (switch-1, 0)
        #     #     c.crz(i, (i + 1) % switch, theta=vqc_weights[b, l, i, 1])
        
        # # 4. Final Hadamard layer (Essential for IQP interference)
        # for i in range(switch):
        #     c.h(i)

        # Strongly Entangling Layer implementation
        for l in range(num_layers):
            # Rotational Part
            for i in range(switch):
                c.r(i, theta=vqc_weights[b, l, i, 0], 
                      alpha=vqc_weights[b, l, i, 1], 
                      phi=vqc_weights[b, l, i, 2])
            # Entanglement Part (Circular CNOTs)
            for i in range(switch):
                c.cnot(i, (i + 1) % switch)

                c.depolarizing(i, px=0.02, py=0.02, pz=0.02, status=0.2)
                c.depolarizing((i + 1) % switch, px=0.02, py=0.02, pz=0.02, status=0.2)

    # # 3. Parity Logic (CNOTs to Ancilla)
    # for i in range(switch):
    #     c.cnot(i, ancilla)

    # for i in range(n_qubits):
    #     c.apply_general_kraus(tc.channels.thermalrelaxationchannel(t1=50, t2=70, time=1), i, status=0.2)
    # for i in range(n_qubits):
    #     c.apply_general_kraus(tc.channels.bitflipchannel(p=0.05), i, status=0.2)

    # 4. Measurement (Cost Type -8)
    # This measures <Z_ancilla * X_switch * product(Z_others)>
    # expectation_ps is very efficient for these long Pauli strings
    
    # Constructing the Pauli string for cost -8:
    # We want <Z(ancilla) X(switch) Z(i)> for specific i
    results = []
    for i in range(switch):
        # expectation_ps uses dictionaries for Pauli operators
        # switch=8, ancilla=9. Remaining data wires are 0-7.
        
        res = c.expectation_ps(x=[switch], z=[j for j in range(switch) if j != i])
        results.append(K.real(res))
    
    # res = c.expectation_ps(x=[switch], z=[ancilla])
    # results.append(K.real(res))
        
    return jnp.array(results)

# JIT for speed
circuit_tc_jit = jax.jit(circuit_tc)