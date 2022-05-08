## DISTRIBUTED CROSS-RELATION-BASED FREQUENCY-DOMAIN BLIND SYSTEM IDENTIFICATION USING ONLINE-ADMM
In this paper, we propose a distributed cross-relation-based adaptive algorithm for blind identification of single-input multiple-output (SIMO) systems in the frequency domain, using the alternating direction method of multipliers (ADMM) in a wireless sensor network (WSN). The network consists of a fixed number of nodes each equipped with a processing unit and a sensor that represents an output channel of the SIMO system. The proposed algorithm exploits the separability of the cross-channel relations by splitting the multichannel identification problem into sub-problems containing a subset of channels, in a way that is determined by the network topology. Each node delivers estimates for the subset of channel frequency responses, which are then combined into a consensus estimate per channel using general-form consensus ADMM in an adaptive updating scheme. Using numerical simulations, we show that it is possible to achieve convergence speeds and steady-state misalignment values comparable to fully centralized low-cost frequency-domain algorithms.

### Repository content
This repository contains all code used to generate the paper submission.
- Python simulation code and plot generation code
  - algorithms: centralized and distributed algorithms used in simulations
  - simulation_1: Random impulse responses, M=5, ring topology
  - simulation_2: Random impulse reponses, M=4, M=8, random topology

### SOUNDS
This research work was carried out at the ESAT Laboratory of KU Leuven, in the frame of the SOUNDS European Training Network.

[SOUNDS Website](https://www.sounds-etn.eu/)

### Acknowledgements
<table>
    <tr>
        <td width="75">
        <img src="https://www.sounds-etn.eu/wp-content/uploads/2021/01/Screenshot-2021-01-07-at-16.50.22-600x400.png"  align="left"/>
        </td>
        <td>
        This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 956369
        </td>
    </tr>
</table>



