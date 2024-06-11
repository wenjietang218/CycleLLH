# CycleLLH
CycleLLH: A New Network Traffic Prediction Model based on Cyclicity Integration
This paper proposes a general network traffic prediction model Cycle Little Linear Head (CycleLLH), which can effectively take advantage of the periodicity. The two key designs of the model are: 1.Cycle Integration: we divide the traffic sequence into different cycle blocks according to a specific period, and then embed the time nodes corresponding to the phase of these cycle blocks into different input tokens; 2.Little Linear Head: it is composed of multiple multi-layer perceptrons, and each multi-layer perceptron operates separately on each feature node.
CycleLLH has achieved the highest traffic prediction accuracy compared with the current latest model, and the traffic prediction accuracy on the three datasets has increased by 12.3%, 8.4%, 29.9%, 5.8%, 8.3% and 2.0% respectively.

# Get Start
1.Install Python>=3.9 and the requirements

2.If you want to replicate the experiments related to CycleLLH in the paper, you can directly run the scripts, which correspond to different experimental results in different folders as described in the paper.
<br />\scripts\traffic_forecasting corresponds to 4.1 Table1 and Table 11
<br />\scripts\longer_look_back_window corresponds to 4.2.3 Figure10
<br />\scripts\different_divison corresponds to 4.3 Figure10
<br />\scripts\with_noise corresponds to 4.5 Table6
<br />\scripts\no_norm corresponds to 4.6 Table7
<br />\scripts\different_cyle corresponds to Appendix B Table8

3.If you want to train this model yourself, please change the default dataset and parameters in run_longExp.py within your scripts.

