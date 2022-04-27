# CROSS-RELATION-BASED FREQUENCY-DOMAIN BLIND SYSTEM IDENTIFICATION USING ONLINE ADMM
In this contribution, we propose a cross-relation-based adaptive algorithm for blind identification of single-input multiple-output (SIMO) systems in the frequency domain using the alternating direction method of multipliers (ADMM).
The proposed algorithm exploits the separability of the cross-channel relations by splitting the multichannel identification problem into lower-dimensional sub-problems with lowered computational complexity, which can be solved in parallel.
Each sub-problem yields estimates for a subset of channel frequency responses, which then are combined into a consensus estimate per channel using general form consensus ADMM in an adaptive updating scheme.
With numerical simulations, we show that it is possible to achieve convergence speeds comparable to low-cost frequency-domain algorithms and estimation errors better than a high-performing Quasi-Newton method.

## Repository content
This repository contains all code used to generate the paper submission.
- LateX code
- Python simulation code

## SOUNDS
This research work was carried out at the ESAT Laboratory of KU Leuven, in the frame of the SOUNDS European Training Network.

[SOUNDS Website](https://www.sounds-etn.eu/)

## Acknowledgements
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



