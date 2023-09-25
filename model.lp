\ Model GNN Inverse
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 C478 + 0 C479 + 0 C494 + 0 C495 + 0 C510 + 0 C511 + 0 C526 + 0 C527
   + 0 C542 + 0 C543 + 0 C558 + 0 C559 + 0 C574 + 0 C575 + 0 C590 + 0 C591
   + Output_t[0] + 5 embedding_similarity
Subject To
 node_0_connected: A[0,1] + A[0,2] + A[0,3] + A[0,4] + A[0,5] + A[0,6]
   + A[0,7] + A[1,0] + A[2,0] + A[3,0] + A[4,0] + A[5,0] + A[6,0] + A[7,0]
   >= 1
 node_1_connected: A[1,2] + A[1,3] + A[1,4] + A[1,5] + A[1,6] + A[1,7]
   + A[2,1] + A[3,1] + A[4,1] + A[5,1] + A[6,1] + A[7,1] >= 1
 node_2_connected: A[2,3] + A[2,4] + A[2,5] + A[2,6] + A[2,7] + A[3,2]
   + A[4,2] + A[5,2] + A[6,2] + A[7,2] >= 1
 node_3_connected: A[3,4] + A[3,5] + A[3,6] + A[3,7] + A[4,3] + A[5,3]
   + A[6,3] + A[7,3] >= 1
 node_4_connected: A[4,5] + A[4,6] + A[4,7] + A[5,4] + A[6,4] + A[7,4]
   >= 1
 node_5_connected: A[5,6] + A[5,7] + A[6,5] + A[7,5] >= 1
 node_6_connected: A[6,7] + A[7,6] >= 1
 undirected_0_0: = 0
 undirected_0_1: A[0,1] - A[1,0] = 0
 undirected_0_2: A[0,2] - A[2,0] = 0
 undirected_0_3: A[0,3] - A[3,0] = 0
 undirected_0_4: A[0,4] - A[4,0] = 0
 undirected_0_5: A[0,5] - A[5,0] = 0
 undirected_0_6: A[0,6] - A[6,0] = 0
 undirected_0_7: A[0,7] - A[7,0] = 0
 undirected_1_1: = 0
 undirected_1_2: A[1,2] - A[2,1] = 0
 undirected_1_3: A[1,3] - A[3,1] = 0
 undirected_1_4: A[1,4] - A[4,1] = 0
 undirected_1_5: A[1,5] - A[5,1] = 0
 undirected_1_6: A[1,6] - A[6,1] = 0
 undirected_1_7: A[1,7] - A[7,1] = 0
 undirected_2_2: = 0
 undirected_2_3: A[2,3] - A[3,2] = 0
 undirected_2_4: A[2,4] - A[4,2] = 0
 undirected_2_5: A[2,5] - A[5,2] = 0
 undirected_2_6: A[2,6] - A[6,2] = 0
 undirected_2_7: A[2,7] - A[7,2] = 0
 undirected_3_3: = 0
 undirected_3_4: A[3,4] - A[4,3] = 0
 undirected_3_5: A[3,5] - A[5,3] = 0
 undirected_3_6: A[3,6] - A[6,3] = 0
 undirected_3_7: A[3,7] - A[7,3] = 0
 undirected_4_4: = 0
 undirected_4_5: A[4,5] - A[5,4] = 0
 undirected_4_6: A[4,6] - A[6,4] = 0
 undirected_4_7: A[4,7] - A[7,4] = 0
 undirected_5_5: = 0
 undirected_5_6: A[5,6] - A[6,5] = 0
 undirected_5_7: A[5,7] - A[7,5] = 0
 undirected_6_6: = 0
 undirected_6_7: A[6,7] - A[7,6] = 0
 undirected_7_7: = 0
 non_isolatied[0]: A[0,0] + A[1,0] + A[2,0] + A[3,0] + A[4,0] + A[5,0]
   + A[6,0] + A[7,0] >= 1
 non_isolatied[1]: A[0,1] + A[1,1] + A[2,1] + A[3,1] + A[4,1] + A[5,1]
   + A[6,1] + A[7,1] >= 1
 non_isolatied[2]: A[0,2] + A[1,2] + A[2,2] + A[3,2] + A[4,2] + A[5,2]
   + A[6,2] + A[7,2] >= 1
 non_isolatied[3]: A[0,3] + A[1,3] + A[2,3] + A[3,3] + A[4,3] + A[5,3]
   + A[6,3] + A[7,3] >= 1
 non_isolatied[4]: A[0,4] + A[1,4] + A[2,4] + A[3,4] + A[4,4] + A[5,4]
   + A[6,4] + A[7,4] >= 1
 non_isolatied[5]: A[0,5] + A[1,5] + A[2,5] + A[3,5] + A[4,5] + A[5,5]
   + A[6,5] + A[7,5] >= 1
 non_isolatied[6]: A[0,6] + A[1,6] + A[2,6] + A[3,6] + A[4,6] + A[5,6]
   + A[6,6] + A[7,6] >= 1
 non_isolatied[7]: A[0,7] + A[1,7] + A[2,7] + A[3,7] + A[4,7] + A[5,7]
   + A[6,7] + A[7,7] >= 1
 features_are_node_degrees[0,0]: - A[0,0] - A[1,0] - A[2,0] - A[3,0]
   - A[4,0] - A[5,0] - A[6,0] - A[7,0] + X[0,0] = 0
 features_are_node_degrees[1,0]: - A[0,1] - A[1,1] - A[2,1] - A[3,1]
   - A[4,1] - A[5,1] - A[6,1] - A[7,1] + X[1,0] = 0
 features_are_node_degrees[2,0]: - A[0,2] - A[1,2] - A[2,2] - A[3,2]
   - A[4,2] - A[5,2] - A[6,2] - A[7,2] + X[2,0] = 0
 features_are_node_degrees[3,0]: - A[0,3] - A[1,3] - A[2,3] - A[3,3]
   - A[4,3] - A[5,3] - A[6,3] - A[7,3] + X[3,0] = 0
 features_are_node_degrees[4,0]: - A[0,4] - A[1,4] - A[2,4] - A[3,4]
   - A[4,4] - A[5,4] - A[6,4] - A[7,4] + X[4,0] = 0
 features_are_node_degrees[5,0]: - A[0,5] - A[1,5] - A[2,5] - A[3,5]
   - A[4,5] - A[5,5] - A[6,5] - A[7,5] + X[5,0] = 0
 features_are_node_degrees[6,0]: - A[0,6] - A[1,6] - A[2,6] - A[3,6]
   - A[4,6] - A[5,6] - A[6,6] - A[7,6] + X[6,0] = 0
 features_are_node_degrees[7,0]: - A[0,7] - A[1,7] - A[2,7] - A[3,7]
   - A[4,7] - A[5,7] - A[6,7] - A[7,7] + X[7,0] = 0
 Conv_0_output_constraint[0,0]: 2.0721332111861557e-04 X[0,0]
   + 0.000222246 Conv_0_aggregated_features[0,0] + Conv_0_t[0,0]
   = -2.39668e-05
 Conv_0_output_constraint[0,1]: - 1.149091243743896 X[0,0]
   + 1.175871133804321 Conv_0_aggregated_features[0,0] + Conv_0_t[0,1]
   = -0.00442547490820289
 Conv_0_output_constraint[0,2]: 0.3711032569408417 X[0,0]
   - 0.3025249838829041 Conv_0_aggregated_features[0,0] + Conv_0_t[0,2]
   = -0.2146252244710922
 Conv_0_output_constraint[0,3]: 5.43493e-09 X[0,0]
   + 5.70055e-08 Conv_0_aggregated_features[0,0] + Conv_0_t[0,3]
   = 1.22294e-14
 Conv_0_output_constraint[0,4]: 2.088812761940062e-04 X[0,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[0,0] + Conv_0_t[0,4]
   = -1.44265e-10
 Conv_0_output_constraint[0,5]: 6.06554e-05 X[0,0]
   + 7.86036e-05 Conv_0_aggregated_features[0,0] + Conv_0_t[0,5]
   = -2.319e-12
 Conv_0_output_constraint[0,6]: 0.0421554557979107 X[0,0]
   + 0.0415282249450684 Conv_0_aggregated_features[0,0] + Conv_0_t[0,6]
   = -1.75046e-05
 Conv_0_output_constraint[0,7]: 1.2457346019800752e-04 X[0,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[0,0] + Conv_0_t[0,7]
   = -5.5964e-09
 Conv_0_output_constraint[0,8]: 7.49032e-08 X[0,0]
   + 7.27417e-08 Conv_0_aggregated_features[0,0] + Conv_0_t[0,8]
   = -5.52095e-08
 Conv_0_output_constraint[0,9]: 1.76828e-10 X[0,0]
   + 1.95361e-10 Conv_0_aggregated_features[0,0] + Conv_0_t[0,9]
   = -9.22955e-11
 Conv_0_output_constraint[0,10]: 0.6364069581031799 X[0,0]
   - 0.4960928857326508 Conv_0_aggregated_features[0,0] + Conv_0_t[0,10]
   = -0.1126708984375
 Conv_0_output_constraint[0,11]: 9.18985e-05 X[0,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[0,0]
   + Conv_0_t[0,11] = -1.58846e-12
 Conv_0_output_constraint[0,12]: - 0.5518577098846436 X[0,0]
   - 0.1993100941181183 Conv_0_aggregated_features[0,0] + Conv_0_t[0,12]
   = -1.310906410217285
 Conv_0_output_constraint[0,13]: 5.59408e-06 X[0,0]
   + 6.37744e-06 Conv_0_aggregated_features[0,0] + Conv_0_t[0,13]
   = 8.63833e-06
 Conv_0_output_constraint[0,14]: Conv_0_t[0,14] = 5.73562e-19
 Conv_0_output_constraint[0,15]: Conv_0_t[0,15] = -1.82729e-42
 Conv_0_output_constraint[1,0]: 2.0721332111861557e-04 X[1,0]
   + 0.000222246 Conv_0_aggregated_features[1,0] + Conv_0_t[1,0]
   = -2.39668e-05
 Conv_0_output_constraint[1,1]: - 1.149091243743896 X[1,0]
   + 1.175871133804321 Conv_0_aggregated_features[1,0] + Conv_0_t[1,1]
   = -0.00442547490820289
 Conv_0_output_constraint[1,2]: 0.3711032569408417 X[1,0]
   - 0.3025249838829041 Conv_0_aggregated_features[1,0] + Conv_0_t[1,2]
   = -0.2146252244710922
 Conv_0_output_constraint[1,3]: 5.43493e-09 X[1,0]
   + 5.70055e-08 Conv_0_aggregated_features[1,0] + Conv_0_t[1,3]
   = 1.22294e-14
 Conv_0_output_constraint[1,4]: 2.088812761940062e-04 X[1,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[1,0] + Conv_0_t[1,4]
   = -1.44265e-10
 Conv_0_output_constraint[1,5]: 6.06554e-05 X[1,0]
   + 7.86036e-05 Conv_0_aggregated_features[1,0] + Conv_0_t[1,5]
   = -2.319e-12
 Conv_0_output_constraint[1,6]: 0.0421554557979107 X[1,0]
   + 0.0415282249450684 Conv_0_aggregated_features[1,0] + Conv_0_t[1,6]
   = -1.75046e-05
 Conv_0_output_constraint[1,7]: 1.2457346019800752e-04 X[1,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[1,0] + Conv_0_t[1,7]
   = -5.5964e-09
 Conv_0_output_constraint[1,8]: 7.49032e-08 X[1,0]
   + 7.27417e-08 Conv_0_aggregated_features[1,0] + Conv_0_t[1,8]
   = -5.52095e-08
 Conv_0_output_constraint[1,9]: 1.76828e-10 X[1,0]
   + 1.95361e-10 Conv_0_aggregated_features[1,0] + Conv_0_t[1,9]
   = -9.22955e-11
 Conv_0_output_constraint[1,10]: 0.6364069581031799 X[1,0]
   - 0.4960928857326508 Conv_0_aggregated_features[1,0] + Conv_0_t[1,10]
   = -0.1126708984375
 Conv_0_output_constraint[1,11]: 9.18985e-05 X[1,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[1,0]
   + Conv_0_t[1,11] = -1.58846e-12
 Conv_0_output_constraint[1,12]: - 0.5518577098846436 X[1,0]
   - 0.1993100941181183 Conv_0_aggregated_features[1,0] + Conv_0_t[1,12]
   = -1.310906410217285
 Conv_0_output_constraint[1,13]: 5.59408e-06 X[1,0]
   + 6.37744e-06 Conv_0_aggregated_features[1,0] + Conv_0_t[1,13]
   = 8.63833e-06
 Conv_0_output_constraint[1,14]: Conv_0_t[1,14] = 5.73562e-19
 Conv_0_output_constraint[1,15]: Conv_0_t[1,15] = -1.82729e-42
 Conv_0_output_constraint[2,0]: 2.0721332111861557e-04 X[2,0]
   + 0.000222246 Conv_0_aggregated_features[2,0] + Conv_0_t[2,0]
   = -2.39668e-05
 Conv_0_output_constraint[2,1]: - 1.149091243743896 X[2,0]
   + 1.175871133804321 Conv_0_aggregated_features[2,0] + Conv_0_t[2,1]
   = -0.00442547490820289
 Conv_0_output_constraint[2,2]: 0.3711032569408417 X[2,0]
   - 0.3025249838829041 Conv_0_aggregated_features[2,0] + Conv_0_t[2,2]
   = -0.2146252244710922
 Conv_0_output_constraint[2,3]: 5.43493e-09 X[2,0]
   + 5.70055e-08 Conv_0_aggregated_features[2,0] + Conv_0_t[2,3]
   = 1.22294e-14
 Conv_0_output_constraint[2,4]: 2.088812761940062e-04 X[2,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[2,0] + Conv_0_t[2,4]
   = -1.44265e-10
 Conv_0_output_constraint[2,5]: 6.06554e-05 X[2,0]
   + 7.86036e-05 Conv_0_aggregated_features[2,0] + Conv_0_t[2,5]
   = -2.319e-12
 Conv_0_output_constraint[2,6]: 0.0421554557979107 X[2,0]
   + 0.0415282249450684 Conv_0_aggregated_features[2,0] + Conv_0_t[2,6]
   = -1.75046e-05
 Conv_0_output_constraint[2,7]: 1.2457346019800752e-04 X[2,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[2,0] + Conv_0_t[2,7]
   = -5.5964e-09
 Conv_0_output_constraint[2,8]: 7.49032e-08 X[2,0]
   + 7.27417e-08 Conv_0_aggregated_features[2,0] + Conv_0_t[2,8]
   = -5.52095e-08
 Conv_0_output_constraint[2,9]: 1.76828e-10 X[2,0]
   + 1.95361e-10 Conv_0_aggregated_features[2,0] + Conv_0_t[2,9]
   = -9.22955e-11
 Conv_0_output_constraint[2,10]: 0.6364069581031799 X[2,0]
   - 0.4960928857326508 Conv_0_aggregated_features[2,0] + Conv_0_t[2,10]
   = -0.1126708984375
 Conv_0_output_constraint[2,11]: 9.18985e-05 X[2,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[2,0]
   + Conv_0_t[2,11] = -1.58846e-12
 Conv_0_output_constraint[2,12]: - 0.5518577098846436 X[2,0]
   - 0.1993100941181183 Conv_0_aggregated_features[2,0] + Conv_0_t[2,12]
   = -1.310906410217285
 Conv_0_output_constraint[2,13]: 5.59408e-06 X[2,0]
   + 6.37744e-06 Conv_0_aggregated_features[2,0] + Conv_0_t[2,13]
   = 8.63833e-06
 Conv_0_output_constraint[2,14]: Conv_0_t[2,14] = 5.73562e-19
 Conv_0_output_constraint[2,15]: Conv_0_t[2,15] = -1.82729e-42
 Conv_0_output_constraint[3,0]: 2.0721332111861557e-04 X[3,0]
   + 0.000222246 Conv_0_aggregated_features[3,0] + Conv_0_t[3,0]
   = -2.39668e-05
 Conv_0_output_constraint[3,1]: - 1.149091243743896 X[3,0]
   + 1.175871133804321 Conv_0_aggregated_features[3,0] + Conv_0_t[3,1]
   = -0.00442547490820289
 Conv_0_output_constraint[3,2]: 0.3711032569408417 X[3,0]
   - 0.3025249838829041 Conv_0_aggregated_features[3,0] + Conv_0_t[3,2]
   = -0.2146252244710922
 Conv_0_output_constraint[3,3]: 5.43493e-09 X[3,0]
   + 5.70055e-08 Conv_0_aggregated_features[3,0] + Conv_0_t[3,3]
   = 1.22294e-14
 Conv_0_output_constraint[3,4]: 2.088812761940062e-04 X[3,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[3,0] + Conv_0_t[3,4]
   = -1.44265e-10
 Conv_0_output_constraint[3,5]: 6.06554e-05 X[3,0]
   + 7.86036e-05 Conv_0_aggregated_features[3,0] + Conv_0_t[3,5]
   = -2.319e-12
 Conv_0_output_constraint[3,6]: 0.0421554557979107 X[3,0]
   + 0.0415282249450684 Conv_0_aggregated_features[3,0] + Conv_0_t[3,6]
   = -1.75046e-05
 Conv_0_output_constraint[3,7]: 1.2457346019800752e-04 X[3,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[3,0] + Conv_0_t[3,7]
   = -5.5964e-09
 Conv_0_output_constraint[3,8]: 7.49032e-08 X[3,0]
   + 7.27417e-08 Conv_0_aggregated_features[3,0] + Conv_0_t[3,8]
   = -5.52095e-08
 Conv_0_output_constraint[3,9]: 1.76828e-10 X[3,0]
   + 1.95361e-10 Conv_0_aggregated_features[3,0] + Conv_0_t[3,9]
   = -9.22955e-11
 Conv_0_output_constraint[3,10]: 0.6364069581031799 X[3,0]
   - 0.4960928857326508 Conv_0_aggregated_features[3,0] + Conv_0_t[3,10]
   = -0.1126708984375
 Conv_0_output_constraint[3,11]: 9.18985e-05 X[3,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[3,0]
   + Conv_0_t[3,11] = -1.58846e-12
 Conv_0_output_constraint[3,12]: - 0.5518577098846436 X[3,0]
   - 0.1993100941181183 Conv_0_aggregated_features[3,0] + Conv_0_t[3,12]
   = -1.310906410217285
 Conv_0_output_constraint[3,13]: 5.59408e-06 X[3,0]
   + 6.37744e-06 Conv_0_aggregated_features[3,0] + Conv_0_t[3,13]
   = 8.63833e-06
 Conv_0_output_constraint[3,14]: Conv_0_t[3,14] = 5.73562e-19
 Conv_0_output_constraint[3,15]: Conv_0_t[3,15] = -1.82729e-42
 Conv_0_output_constraint[4,0]: 2.0721332111861557e-04 X[4,0]
   + 0.000222246 Conv_0_aggregated_features[4,0] + Conv_0_t[4,0]
   = -2.39668e-05
 Conv_0_output_constraint[4,1]: - 1.149091243743896 X[4,0]
   + 1.175871133804321 Conv_0_aggregated_features[4,0] + Conv_0_t[4,1]
   = -0.00442547490820289
 Conv_0_output_constraint[4,2]: 0.3711032569408417 X[4,0]
   - 0.3025249838829041 Conv_0_aggregated_features[4,0] + Conv_0_t[4,2]
   = -0.2146252244710922
 Conv_0_output_constraint[4,3]: 5.43493e-09 X[4,0]
   + 5.70055e-08 Conv_0_aggregated_features[4,0] + Conv_0_t[4,3]
   = 1.22294e-14
 Conv_0_output_constraint[4,4]: 2.088812761940062e-04 X[4,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[4,0] + Conv_0_t[4,4]
   = -1.44265e-10
 Conv_0_output_constraint[4,5]: 6.06554e-05 X[4,0]
   + 7.86036e-05 Conv_0_aggregated_features[4,0] + Conv_0_t[4,5]
   = -2.319e-12
 Conv_0_output_constraint[4,6]: 0.0421554557979107 X[4,0]
   + 0.0415282249450684 Conv_0_aggregated_features[4,0] + Conv_0_t[4,6]
   = -1.75046e-05
 Conv_0_output_constraint[4,7]: 1.2457346019800752e-04 X[4,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[4,0] + Conv_0_t[4,7]
   = -5.5964e-09
 Conv_0_output_constraint[4,8]: 7.49032e-08 X[4,0]
   + 7.27417e-08 Conv_0_aggregated_features[4,0] + Conv_0_t[4,8]
   = -5.52095e-08
 Conv_0_output_constraint[4,9]: 1.76828e-10 X[4,0]
   + 1.95361e-10 Conv_0_aggregated_features[4,0] + Conv_0_t[4,9]
   = -9.22955e-11
 Conv_0_output_constraint[4,10]: 0.6364069581031799 X[4,0]
   - 0.4960928857326508 Conv_0_aggregated_features[4,0] + Conv_0_t[4,10]
   = -0.1126708984375
 Conv_0_output_constraint[4,11]: 9.18985e-05 X[4,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[4,0]
   + Conv_0_t[4,11] = -1.58846e-12
 Conv_0_output_constraint[4,12]: - 0.5518577098846436 X[4,0]
   - 0.1993100941181183 Conv_0_aggregated_features[4,0] + Conv_0_t[4,12]
   = -1.310906410217285
 Conv_0_output_constraint[4,13]: 5.59408e-06 X[4,0]
   + 6.37744e-06 Conv_0_aggregated_features[4,0] + Conv_0_t[4,13]
   = 8.63833e-06
 Conv_0_output_constraint[4,14]: Conv_0_t[4,14] = 5.73562e-19
 Conv_0_output_constraint[4,15]: Conv_0_t[4,15] = -1.82729e-42
 Conv_0_output_constraint[5,0]: 2.0721332111861557e-04 X[5,0]
   + 0.000222246 Conv_0_aggregated_features[5,0] + Conv_0_t[5,0]
   = -2.39668e-05
 Conv_0_output_constraint[5,1]: - 1.149091243743896 X[5,0]
   + 1.175871133804321 Conv_0_aggregated_features[5,0] + Conv_0_t[5,1]
   = -0.00442547490820289
 Conv_0_output_constraint[5,2]: 0.3711032569408417 X[5,0]
   - 0.3025249838829041 Conv_0_aggregated_features[5,0] + Conv_0_t[5,2]
   = -0.2146252244710922
 Conv_0_output_constraint[5,3]: 5.43493e-09 X[5,0]
   + 5.70055e-08 Conv_0_aggregated_features[5,0] + Conv_0_t[5,3]
   = 1.22294e-14
 Conv_0_output_constraint[5,4]: 2.088812761940062e-04 X[5,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[5,0] + Conv_0_t[5,4]
   = -1.44265e-10
 Conv_0_output_constraint[5,5]: 6.06554e-05 X[5,0]
   + 7.86036e-05 Conv_0_aggregated_features[5,0] + Conv_0_t[5,5]
   = -2.319e-12
 Conv_0_output_constraint[5,6]: 0.0421554557979107 X[5,0]
   + 0.0415282249450684 Conv_0_aggregated_features[5,0] + Conv_0_t[5,6]
   = -1.75046e-05
 Conv_0_output_constraint[5,7]: 1.2457346019800752e-04 X[5,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[5,0] + Conv_0_t[5,7]
   = -5.5964e-09
 Conv_0_output_constraint[5,8]: 7.49032e-08 X[5,0]
   + 7.27417e-08 Conv_0_aggregated_features[5,0] + Conv_0_t[5,8]
   = -5.52095e-08
 Conv_0_output_constraint[5,9]: 1.76828e-10 X[5,0]
   + 1.95361e-10 Conv_0_aggregated_features[5,0] + Conv_0_t[5,9]
   = -9.22955e-11
 Conv_0_output_constraint[5,10]: 0.6364069581031799 X[5,0]
   - 0.4960928857326508 Conv_0_aggregated_features[5,0] + Conv_0_t[5,10]
   = -0.1126708984375
 Conv_0_output_constraint[5,11]: 9.18985e-05 X[5,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[5,0]
   + Conv_0_t[5,11] = -1.58846e-12
 Conv_0_output_constraint[5,12]: - 0.5518577098846436 X[5,0]
   - 0.1993100941181183 Conv_0_aggregated_features[5,0] + Conv_0_t[5,12]
   = -1.310906410217285
 Conv_0_output_constraint[5,13]: 5.59408e-06 X[5,0]
   + 6.37744e-06 Conv_0_aggregated_features[5,0] + Conv_0_t[5,13]
   = 8.63833e-06
 Conv_0_output_constraint[5,14]: Conv_0_t[5,14] = 5.73562e-19
 Conv_0_output_constraint[5,15]: Conv_0_t[5,15] = -1.82729e-42
 Conv_0_output_constraint[6,0]: 2.0721332111861557e-04 X[6,0]
   + 0.000222246 Conv_0_aggregated_features[6,0] + Conv_0_t[6,0]
   = -2.39668e-05
 Conv_0_output_constraint[6,1]: - 1.149091243743896 X[6,0]
   + 1.175871133804321 Conv_0_aggregated_features[6,0] + Conv_0_t[6,1]
   = -0.00442547490820289
 Conv_0_output_constraint[6,2]: 0.3711032569408417 X[6,0]
   - 0.3025249838829041 Conv_0_aggregated_features[6,0] + Conv_0_t[6,2]
   = -0.2146252244710922
 Conv_0_output_constraint[6,3]: 5.43493e-09 X[6,0]
   + 5.70055e-08 Conv_0_aggregated_features[6,0] + Conv_0_t[6,3]
   = 1.22294e-14
 Conv_0_output_constraint[6,4]: 2.088812761940062e-04 X[6,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[6,0] + Conv_0_t[6,4]
   = -1.44265e-10
 Conv_0_output_constraint[6,5]: 6.06554e-05 X[6,0]
   + 7.86036e-05 Conv_0_aggregated_features[6,0] + Conv_0_t[6,5]
   = -2.319e-12
 Conv_0_output_constraint[6,6]: 0.0421554557979107 X[6,0]
   + 0.0415282249450684 Conv_0_aggregated_features[6,0] + Conv_0_t[6,6]
   = -1.75046e-05
 Conv_0_output_constraint[6,7]: 1.2457346019800752e-04 X[6,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[6,0] + Conv_0_t[6,7]
   = -5.5964e-09
 Conv_0_output_constraint[6,8]: 7.49032e-08 X[6,0]
   + 7.27417e-08 Conv_0_aggregated_features[6,0] + Conv_0_t[6,8]
   = -5.52095e-08
 Conv_0_output_constraint[6,9]: 1.76828e-10 X[6,0]
   + 1.95361e-10 Conv_0_aggregated_features[6,0] + Conv_0_t[6,9]
   = -9.22955e-11
 Conv_0_output_constraint[6,10]: 0.6364069581031799 X[6,0]
   - 0.4960928857326508 Conv_0_aggregated_features[6,0] + Conv_0_t[6,10]
   = -0.1126708984375
 Conv_0_output_constraint[6,11]: 9.18985e-05 X[6,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[6,0]
   + Conv_0_t[6,11] = -1.58846e-12
 Conv_0_output_constraint[6,12]: - 0.5518577098846436 X[6,0]
   - 0.1993100941181183 Conv_0_aggregated_features[6,0] + Conv_0_t[6,12]
   = -1.310906410217285
 Conv_0_output_constraint[6,13]: 5.59408e-06 X[6,0]
   + 6.37744e-06 Conv_0_aggregated_features[6,0] + Conv_0_t[6,13]
   = 8.63833e-06
 Conv_0_output_constraint[6,14]: Conv_0_t[6,14] = 5.73562e-19
 Conv_0_output_constraint[6,15]: Conv_0_t[6,15] = -1.82729e-42
 Conv_0_output_constraint[7,0]: 2.0721332111861557e-04 X[7,0]
   + 0.000222246 Conv_0_aggregated_features[7,0] + Conv_0_t[7,0]
   = -2.39668e-05
 Conv_0_output_constraint[7,1]: - 1.149091243743896 X[7,0]
   + 1.175871133804321 Conv_0_aggregated_features[7,0] + Conv_0_t[7,1]
   = -0.00442547490820289
 Conv_0_output_constraint[7,2]: 0.3711032569408417 X[7,0]
   - 0.3025249838829041 Conv_0_aggregated_features[7,0] + Conv_0_t[7,2]
   = -0.2146252244710922
 Conv_0_output_constraint[7,3]: 5.43493e-09 X[7,0]
   + 5.70055e-08 Conv_0_aggregated_features[7,0] + Conv_0_t[7,3]
   = 1.22294e-14
 Conv_0_output_constraint[7,4]: 2.088812761940062e-04 X[7,0]
   + 2.0406844851095229e-04 Conv_0_aggregated_features[7,0] + Conv_0_t[7,4]
   = -1.44265e-10
 Conv_0_output_constraint[7,5]: 6.06554e-05 X[7,0]
   + 7.86036e-05 Conv_0_aggregated_features[7,0] + Conv_0_t[7,5]
   = -2.319e-12
 Conv_0_output_constraint[7,6]: 0.0421554557979107 X[7,0]
   + 0.0415282249450684 Conv_0_aggregated_features[7,0] + Conv_0_t[7,6]
   = -1.75046e-05
 Conv_0_output_constraint[7,7]: 1.2457346019800752e-04 X[7,0]
   + 7.7460560714825988e-04 Conv_0_aggregated_features[7,0] + Conv_0_t[7,7]
   = -5.5964e-09
 Conv_0_output_constraint[7,8]: 7.49032e-08 X[7,0]
   + 7.27417e-08 Conv_0_aggregated_features[7,0] + Conv_0_t[7,8]
   = -5.52095e-08
 Conv_0_output_constraint[7,9]: 1.76828e-10 X[7,0]
   + 1.95361e-10 Conv_0_aggregated_features[7,0] + Conv_0_t[7,9]
   = -9.22955e-11
 Conv_0_output_constraint[7,10]: 0.6364069581031799 X[7,0]
   - 0.4960928857326508 Conv_0_aggregated_features[7,0] + Conv_0_t[7,10]
   = -0.1126708984375
 Conv_0_output_constraint[7,11]: 9.18985e-05 X[7,0]
   + 1.8595183792058378e-04 Conv_0_aggregated_features[7,0]
   + Conv_0_t[7,11] = -1.58846e-12
 Conv_0_output_constraint[7,12]: - 0.5518577098846436 X[7,0]
   - 0.1993100941181183 Conv_0_aggregated_features[7,0] + Conv_0_t[7,12]
   = -1.310906410217285
 Conv_0_output_constraint[7,13]: 5.59408e-06 X[7,0]
   + 6.37744e-06 Conv_0_aggregated_features[7,0] + Conv_0_t[7,13]
   = 8.63833e-06
 Conv_0_output_constraint[7,14]: Conv_0_t[7,14] = 5.73562e-19
 Conv_0_output_constraint[7,15]: Conv_0_t[7,15] = -1.82729e-42
 Conv_0_Relu_constraint_1[0,0]: - Conv_0_t[0,0] + Conv_0_Relu_ts[0,0]
   - Conv_0_Relu_ss[0,0] = 0
 Conv_0_Relu_constraint_1[0,1]: - Conv_0_t[0,1] + Conv_0_Relu_ts[0,1]
   - Conv_0_Relu_ss[0,1] = 0
 Conv_0_Relu_constraint_1[0,2]: - Conv_0_t[0,2] + Conv_0_Relu_ts[0,2]
   - Conv_0_Relu_ss[0,2] = 0
 Conv_0_Relu_constraint_1[0,3]: - Conv_0_t[0,3] + Conv_0_Relu_ts[0,3]
   - Conv_0_Relu_ss[0,3] = 0
 Conv_0_Relu_constraint_1[0,4]: - Conv_0_t[0,4] + Conv_0_Relu_ts[0,4]
   - Conv_0_Relu_ss[0,4] = 0
 Conv_0_Relu_constraint_1[0,5]: - Conv_0_t[0,5] + Conv_0_Relu_ts[0,5]
   - Conv_0_Relu_ss[0,5] = 0
 Conv_0_Relu_constraint_1[0,6]: - Conv_0_t[0,6] + Conv_0_Relu_ts[0,6]
   - Conv_0_Relu_ss[0,6] = 0
 Conv_0_Relu_constraint_1[0,7]: - Conv_0_t[0,7] + Conv_0_Relu_ts[0,7]
   - Conv_0_Relu_ss[0,7] = 0
 Conv_0_Relu_constraint_1[0,8]: - Conv_0_t[0,8] + Conv_0_Relu_ts[0,8]
   - Conv_0_Relu_ss[0,8] = 0
 Conv_0_Relu_constraint_1[0,9]: - Conv_0_t[0,9] + Conv_0_Relu_ts[0,9]
   - Conv_0_Relu_ss[0,9] = 0
 Conv_0_Relu_constraint_1[0,10]: - Conv_0_t[0,10] + Conv_0_Relu_ts[0,10]
   - Conv_0_Relu_ss[0,10] = 0
 Conv_0_Relu_constraint_1[0,11]: - Conv_0_t[0,11] + Conv_0_Relu_ts[0,11]
   - Conv_0_Relu_ss[0,11] = 0
 Conv_0_Relu_constraint_1[0,12]: - Conv_0_t[0,12] + Conv_0_Relu_ts[0,12]
   - Conv_0_Relu_ss[0,12] = 0
 Conv_0_Relu_constraint_1[0,13]: - Conv_0_t[0,13] + Conv_0_Relu_ts[0,13]
   - Conv_0_Relu_ss[0,13] = 0
 Conv_0_Relu_constraint_1[0,14]: - Conv_0_t[0,14] + Conv_0_Relu_ts[0,14]
   - Conv_0_Relu_ss[0,14] = 0
 Conv_0_Relu_constraint_1[0,15]: - Conv_0_t[0,15] + Conv_0_Relu_ts[0,15]
   - Conv_0_Relu_ss[0,15] = 0
 Conv_0_Relu_constraint_1[1,0]: - Conv_0_t[1,0] + Conv_0_Relu_ts[1,0]
   - Conv_0_Relu_ss[1,0] = 0
 Conv_0_Relu_constraint_1[1,1]: - Conv_0_t[1,1] + Conv_0_Relu_ts[1,1]
   - Conv_0_Relu_ss[1,1] = 0
 Conv_0_Relu_constraint_1[1,2]: - Conv_0_t[1,2] + Conv_0_Relu_ts[1,2]
   - Conv_0_Relu_ss[1,2] = 0
 Conv_0_Relu_constraint_1[1,3]: - Conv_0_t[1,3] + Conv_0_Relu_ts[1,3]
   - Conv_0_Relu_ss[1,3] = 0
 Conv_0_Relu_constraint_1[1,4]: - Conv_0_t[1,4] + Conv_0_Relu_ts[1,4]
   - Conv_0_Relu_ss[1,4] = 0
 Conv_0_Relu_constraint_1[1,5]: - Conv_0_t[1,5] + Conv_0_Relu_ts[1,5]
   - Conv_0_Relu_ss[1,5] = 0
 Conv_0_Relu_constraint_1[1,6]: - Conv_0_t[1,6] + Conv_0_Relu_ts[1,6]
   - Conv_0_Relu_ss[1,6] = 0
 Conv_0_Relu_constraint_1[1,7]: - Conv_0_t[1,7] + Conv_0_Relu_ts[1,7]
   - Conv_0_Relu_ss[1,7] = 0
 Conv_0_Relu_constraint_1[1,8]: - Conv_0_t[1,8] + Conv_0_Relu_ts[1,8]
   - Conv_0_Relu_ss[1,8] = 0
 Conv_0_Relu_constraint_1[1,9]: - Conv_0_t[1,9] + Conv_0_Relu_ts[1,9]
   - Conv_0_Relu_ss[1,9] = 0
 Conv_0_Relu_constraint_1[1,10]: - Conv_0_t[1,10] + Conv_0_Relu_ts[1,10]
   - Conv_0_Relu_ss[1,10] = 0
 Conv_0_Relu_constraint_1[1,11]: - Conv_0_t[1,11] + Conv_0_Relu_ts[1,11]
   - Conv_0_Relu_ss[1,11] = 0
 Conv_0_Relu_constraint_1[1,12]: - Conv_0_t[1,12] + Conv_0_Relu_ts[1,12]
   - Conv_0_Relu_ss[1,12] = 0
 Conv_0_Relu_constraint_1[1,13]: - Conv_0_t[1,13] + Conv_0_Relu_ts[1,13]
   - Conv_0_Relu_ss[1,13] = 0
 Conv_0_Relu_constraint_1[1,14]: - Conv_0_t[1,14] + Conv_0_Relu_ts[1,14]
   - Conv_0_Relu_ss[1,14] = 0
 Conv_0_Relu_constraint_1[1,15]: - Conv_0_t[1,15] + Conv_0_Relu_ts[1,15]
   - Conv_0_Relu_ss[1,15] = 0
 Conv_0_Relu_constraint_1[2,0]: - Conv_0_t[2,0] + Conv_0_Relu_ts[2,0]
   - Conv_0_Relu_ss[2,0] = 0
 Conv_0_Relu_constraint_1[2,1]: - Conv_0_t[2,1] + Conv_0_Relu_ts[2,1]
   - Conv_0_Relu_ss[2,1] = 0
 Conv_0_Relu_constraint_1[2,2]: - Conv_0_t[2,2] + Conv_0_Relu_ts[2,2]
   - Conv_0_Relu_ss[2,2] = 0
 Conv_0_Relu_constraint_1[2,3]: - Conv_0_t[2,3] + Conv_0_Relu_ts[2,3]
   - Conv_0_Relu_ss[2,3] = 0
 Conv_0_Relu_constraint_1[2,4]: - Conv_0_t[2,4] + Conv_0_Relu_ts[2,4]
   - Conv_0_Relu_ss[2,4] = 0
 Conv_0_Relu_constraint_1[2,5]: - Conv_0_t[2,5] + Conv_0_Relu_ts[2,5]
   - Conv_0_Relu_ss[2,5] = 0
 Conv_0_Relu_constraint_1[2,6]: - Conv_0_t[2,6] + Conv_0_Relu_ts[2,6]
   - Conv_0_Relu_ss[2,6] = 0
 Conv_0_Relu_constraint_1[2,7]: - Conv_0_t[2,7] + Conv_0_Relu_ts[2,7]
   - Conv_0_Relu_ss[2,7] = 0
 Conv_0_Relu_constraint_1[2,8]: - Conv_0_t[2,8] + Conv_0_Relu_ts[2,8]
   - Conv_0_Relu_ss[2,8] = 0
 Conv_0_Relu_constraint_1[2,9]: - Conv_0_t[2,9] + Conv_0_Relu_ts[2,9]
   - Conv_0_Relu_ss[2,9] = 0
 Conv_0_Relu_constraint_1[2,10]: - Conv_0_t[2,10] + Conv_0_Relu_ts[2,10]
   - Conv_0_Relu_ss[2,10] = 0
 Conv_0_Relu_constraint_1[2,11]: - Conv_0_t[2,11] + Conv_0_Relu_ts[2,11]
   - Conv_0_Relu_ss[2,11] = 0
 Conv_0_Relu_constraint_1[2,12]: - Conv_0_t[2,12] + Conv_0_Relu_ts[2,12]
   - Conv_0_Relu_ss[2,12] = 0
 Conv_0_Relu_constraint_1[2,13]: - Conv_0_t[2,13] + Conv_0_Relu_ts[2,13]
   - Conv_0_Relu_ss[2,13] = 0
 Conv_0_Relu_constraint_1[2,14]: - Conv_0_t[2,14] + Conv_0_Relu_ts[2,14]
   - Conv_0_Relu_ss[2,14] = 0
 Conv_0_Relu_constraint_1[2,15]: - Conv_0_t[2,15] + Conv_0_Relu_ts[2,15]
   - Conv_0_Relu_ss[2,15] = 0
 Conv_0_Relu_constraint_1[3,0]: - Conv_0_t[3,0] + Conv_0_Relu_ts[3,0]
   - Conv_0_Relu_ss[3,0] = 0
 Conv_0_Relu_constraint_1[3,1]: - Conv_0_t[3,1] + Conv_0_Relu_ts[3,1]
   - Conv_0_Relu_ss[3,1] = 0
 Conv_0_Relu_constraint_1[3,2]: - Conv_0_t[3,2] + Conv_0_Relu_ts[3,2]
   - Conv_0_Relu_ss[3,2] = 0
 Conv_0_Relu_constraint_1[3,3]: - Conv_0_t[3,3] + Conv_0_Relu_ts[3,3]
   - Conv_0_Relu_ss[3,3] = 0
 Conv_0_Relu_constraint_1[3,4]: - Conv_0_t[3,4] + Conv_0_Relu_ts[3,4]
   - Conv_0_Relu_ss[3,4] = 0
 Conv_0_Relu_constraint_1[3,5]: - Conv_0_t[3,5] + Conv_0_Relu_ts[3,5]
   - Conv_0_Relu_ss[3,5] = 0
 Conv_0_Relu_constraint_1[3,6]: - Conv_0_t[3,6] + Conv_0_Relu_ts[3,6]
   - Conv_0_Relu_ss[3,6] = 0
 Conv_0_Relu_constraint_1[3,7]: - Conv_0_t[3,7] + Conv_0_Relu_ts[3,7]
   - Conv_0_Relu_ss[3,7] = 0
 Conv_0_Relu_constraint_1[3,8]: - Conv_0_t[3,8] + Conv_0_Relu_ts[3,8]
   - Conv_0_Relu_ss[3,8] = 0
 Conv_0_Relu_constraint_1[3,9]: - Conv_0_t[3,9] + Conv_0_Relu_ts[3,9]
   - Conv_0_Relu_ss[3,9] = 0
 Conv_0_Relu_constraint_1[3,10]: - Conv_0_t[3,10] + Conv_0_Relu_ts[3,10]
   - Conv_0_Relu_ss[3,10] = 0
 Conv_0_Relu_constraint_1[3,11]: - Conv_0_t[3,11] + Conv_0_Relu_ts[3,11]
   - Conv_0_Relu_ss[3,11] = 0
 Conv_0_Relu_constraint_1[3,12]: - Conv_0_t[3,12] + Conv_0_Relu_ts[3,12]
   - Conv_0_Relu_ss[3,12] = 0
 Conv_0_Relu_constraint_1[3,13]: - Conv_0_t[3,13] + Conv_0_Relu_ts[3,13]
   - Conv_0_Relu_ss[3,13] = 0
 Conv_0_Relu_constraint_1[3,14]: - Conv_0_t[3,14] + Conv_0_Relu_ts[3,14]
   - Conv_0_Relu_ss[3,14] = 0
 Conv_0_Relu_constraint_1[3,15]: - Conv_0_t[3,15] + Conv_0_Relu_ts[3,15]
   - Conv_0_Relu_ss[3,15] = 0
 Conv_0_Relu_constraint_1[4,0]: - Conv_0_t[4,0] + Conv_0_Relu_ts[4,0]
   - Conv_0_Relu_ss[4,0] = 0
 Conv_0_Relu_constraint_1[4,1]: - Conv_0_t[4,1] + Conv_0_Relu_ts[4,1]
   - Conv_0_Relu_ss[4,1] = 0
 Conv_0_Relu_constraint_1[4,2]: - Conv_0_t[4,2] + Conv_0_Relu_ts[4,2]
   - Conv_0_Relu_ss[4,2] = 0
 Conv_0_Relu_constraint_1[4,3]: - Conv_0_t[4,3] + Conv_0_Relu_ts[4,3]
   - Conv_0_Relu_ss[4,3] = 0
 Conv_0_Relu_constraint_1[4,4]: - Conv_0_t[4,4] + Conv_0_Relu_ts[4,4]
   - Conv_0_Relu_ss[4,4] = 0
 Conv_0_Relu_constraint_1[4,5]: - Conv_0_t[4,5] + Conv_0_Relu_ts[4,5]
   - Conv_0_Relu_ss[4,5] = 0
 Conv_0_Relu_constraint_1[4,6]: - Conv_0_t[4,6] + Conv_0_Relu_ts[4,6]
   - Conv_0_Relu_ss[4,6] = 0
 Conv_0_Relu_constraint_1[4,7]: - Conv_0_t[4,7] + Conv_0_Relu_ts[4,7]
   - Conv_0_Relu_ss[4,7] = 0
 Conv_0_Relu_constraint_1[4,8]: - Conv_0_t[4,8] + Conv_0_Relu_ts[4,8]
   - Conv_0_Relu_ss[4,8] = 0
 Conv_0_Relu_constraint_1[4,9]: - Conv_0_t[4,9] + Conv_0_Relu_ts[4,9]
   - Conv_0_Relu_ss[4,9] = 0
 Conv_0_Relu_constraint_1[4,10]: - Conv_0_t[4,10] + Conv_0_Relu_ts[4,10]
   - Conv_0_Relu_ss[4,10] = 0
 Conv_0_Relu_constraint_1[4,11]: - Conv_0_t[4,11] + Conv_0_Relu_ts[4,11]
   - Conv_0_Relu_ss[4,11] = 0
 Conv_0_Relu_constraint_1[4,12]: - Conv_0_t[4,12] + Conv_0_Relu_ts[4,12]
   - Conv_0_Relu_ss[4,12] = 0
 Conv_0_Relu_constraint_1[4,13]: - Conv_0_t[4,13] + Conv_0_Relu_ts[4,13]
   - Conv_0_Relu_ss[4,13] = 0
 Conv_0_Relu_constraint_1[4,14]: - Conv_0_t[4,14] + Conv_0_Relu_ts[4,14]
   - Conv_0_Relu_ss[4,14] = 0
 Conv_0_Relu_constraint_1[4,15]: - Conv_0_t[4,15] + Conv_0_Relu_ts[4,15]
   - Conv_0_Relu_ss[4,15] = 0
 Conv_0_Relu_constraint_1[5,0]: - Conv_0_t[5,0] + Conv_0_Relu_ts[5,0]
   - Conv_0_Relu_ss[5,0] = 0
 Conv_0_Relu_constraint_1[5,1]: - Conv_0_t[5,1] + Conv_0_Relu_ts[5,1]
   - Conv_0_Relu_ss[5,1] = 0
 Conv_0_Relu_constraint_1[5,2]: - Conv_0_t[5,2] + Conv_0_Relu_ts[5,2]
   - Conv_0_Relu_ss[5,2] = 0
 Conv_0_Relu_constraint_1[5,3]: - Conv_0_t[5,3] + Conv_0_Relu_ts[5,3]
   - Conv_0_Relu_ss[5,3] = 0
 Conv_0_Relu_constraint_1[5,4]: - Conv_0_t[5,4] + Conv_0_Relu_ts[5,4]
   - Conv_0_Relu_ss[5,4] = 0
 Conv_0_Relu_constraint_1[5,5]: - Conv_0_t[5,5] + Conv_0_Relu_ts[5,5]
   - Conv_0_Relu_ss[5,5] = 0
 Conv_0_Relu_constraint_1[5,6]: - Conv_0_t[5,6] + Conv_0_Relu_ts[5,6]
   - Conv_0_Relu_ss[5,6] = 0
 Conv_0_Relu_constraint_1[5,7]: - Conv_0_t[5,7] + Conv_0_Relu_ts[5,7]
   - Conv_0_Relu_ss[5,7] = 0
 Conv_0_Relu_constraint_1[5,8]: - Conv_0_t[5,8] + Conv_0_Relu_ts[5,8]
   - Conv_0_Relu_ss[5,8] = 0
 Conv_0_Relu_constraint_1[5,9]: - Conv_0_t[5,9] + Conv_0_Relu_ts[5,9]
   - Conv_0_Relu_ss[5,9] = 0
 Conv_0_Relu_constraint_1[5,10]: - Conv_0_t[5,10] + Conv_0_Relu_ts[5,10]
   - Conv_0_Relu_ss[5,10] = 0
 Conv_0_Relu_constraint_1[5,11]: - Conv_0_t[5,11] + Conv_0_Relu_ts[5,11]
   - Conv_0_Relu_ss[5,11] = 0
 Conv_0_Relu_constraint_1[5,12]: - Conv_0_t[5,12] + Conv_0_Relu_ts[5,12]
   - Conv_0_Relu_ss[5,12] = 0
 Conv_0_Relu_constraint_1[5,13]: - Conv_0_t[5,13] + Conv_0_Relu_ts[5,13]
   - Conv_0_Relu_ss[5,13] = 0
 Conv_0_Relu_constraint_1[5,14]: - Conv_0_t[5,14] + Conv_0_Relu_ts[5,14]
   - Conv_0_Relu_ss[5,14] = 0
 Conv_0_Relu_constraint_1[5,15]: - Conv_0_t[5,15] + Conv_0_Relu_ts[5,15]
   - Conv_0_Relu_ss[5,15] = 0
 Conv_0_Relu_constraint_1[6,0]: - Conv_0_t[6,0] + Conv_0_Relu_ts[6,0]
   - Conv_0_Relu_ss[6,0] = 0
 Conv_0_Relu_constraint_1[6,1]: - Conv_0_t[6,1] + Conv_0_Relu_ts[6,1]
   - Conv_0_Relu_ss[6,1] = 0
 Conv_0_Relu_constraint_1[6,2]: - Conv_0_t[6,2] + Conv_0_Relu_ts[6,2]
   - Conv_0_Relu_ss[6,2] = 0
 Conv_0_Relu_constraint_1[6,3]: - Conv_0_t[6,3] + Conv_0_Relu_ts[6,3]
   - Conv_0_Relu_ss[6,3] = 0
 Conv_0_Relu_constraint_1[6,4]: - Conv_0_t[6,4] + Conv_0_Relu_ts[6,4]
   - Conv_0_Relu_ss[6,4] = 0
 Conv_0_Relu_constraint_1[6,5]: - Conv_0_t[6,5] + Conv_0_Relu_ts[6,5]
   - Conv_0_Relu_ss[6,5] = 0
 Conv_0_Relu_constraint_1[6,6]: - Conv_0_t[6,6] + Conv_0_Relu_ts[6,6]
   - Conv_0_Relu_ss[6,6] = 0
 Conv_0_Relu_constraint_1[6,7]: - Conv_0_t[6,7] + Conv_0_Relu_ts[6,7]
   - Conv_0_Relu_ss[6,7] = 0
 Conv_0_Relu_constraint_1[6,8]: - Conv_0_t[6,8] + Conv_0_Relu_ts[6,8]
   - Conv_0_Relu_ss[6,8] = 0
 Conv_0_Relu_constraint_1[6,9]: - Conv_0_t[6,9] + Conv_0_Relu_ts[6,9]
   - Conv_0_Relu_ss[6,9] = 0
 Conv_0_Relu_constraint_1[6,10]: - Conv_0_t[6,10] + Conv_0_Relu_ts[6,10]
   - Conv_0_Relu_ss[6,10] = 0
 Conv_0_Relu_constraint_1[6,11]: - Conv_0_t[6,11] + Conv_0_Relu_ts[6,11]
   - Conv_0_Relu_ss[6,11] = 0
 Conv_0_Relu_constraint_1[6,12]: - Conv_0_t[6,12] + Conv_0_Relu_ts[6,12]
   - Conv_0_Relu_ss[6,12] = 0
 Conv_0_Relu_constraint_1[6,13]: - Conv_0_t[6,13] + Conv_0_Relu_ts[6,13]
   - Conv_0_Relu_ss[6,13] = 0
 Conv_0_Relu_constraint_1[6,14]: - Conv_0_t[6,14] + Conv_0_Relu_ts[6,14]
   - Conv_0_Relu_ss[6,14] = 0
 Conv_0_Relu_constraint_1[6,15]: - Conv_0_t[6,15] + Conv_0_Relu_ts[6,15]
   - Conv_0_Relu_ss[6,15] = 0
 Conv_0_Relu_constraint_1[7,0]: - Conv_0_t[7,0] + Conv_0_Relu_ts[7,0]
   - Conv_0_Relu_ss[7,0] = 0
 Conv_0_Relu_constraint_1[7,1]: - Conv_0_t[7,1] + Conv_0_Relu_ts[7,1]
   - Conv_0_Relu_ss[7,1] = 0
 Conv_0_Relu_constraint_1[7,2]: - Conv_0_t[7,2] + Conv_0_Relu_ts[7,2]
   - Conv_0_Relu_ss[7,2] = 0
 Conv_0_Relu_constraint_1[7,3]: - Conv_0_t[7,3] + Conv_0_Relu_ts[7,3]
   - Conv_0_Relu_ss[7,3] = 0
 Conv_0_Relu_constraint_1[7,4]: - Conv_0_t[7,4] + Conv_0_Relu_ts[7,4]
   - Conv_0_Relu_ss[7,4] = 0
 Conv_0_Relu_constraint_1[7,5]: - Conv_0_t[7,5] + Conv_0_Relu_ts[7,5]
   - Conv_0_Relu_ss[7,5] = 0
 Conv_0_Relu_constraint_1[7,6]: - Conv_0_t[7,6] + Conv_0_Relu_ts[7,6]
   - Conv_0_Relu_ss[7,6] = 0
 Conv_0_Relu_constraint_1[7,7]: - Conv_0_t[7,7] + Conv_0_Relu_ts[7,7]
   - Conv_0_Relu_ss[7,7] = 0
 Conv_0_Relu_constraint_1[7,8]: - Conv_0_t[7,8] + Conv_0_Relu_ts[7,8]
   - Conv_0_Relu_ss[7,8] = 0
 Conv_0_Relu_constraint_1[7,9]: - Conv_0_t[7,9] + Conv_0_Relu_ts[7,9]
   - Conv_0_Relu_ss[7,9] = 0
 Conv_0_Relu_constraint_1[7,10]: - Conv_0_t[7,10] + Conv_0_Relu_ts[7,10]
   - Conv_0_Relu_ss[7,10] = 0
 Conv_0_Relu_constraint_1[7,11]: - Conv_0_t[7,11] + Conv_0_Relu_ts[7,11]
   - Conv_0_Relu_ss[7,11] = 0
 Conv_0_Relu_constraint_1[7,12]: - Conv_0_t[7,12] + Conv_0_Relu_ts[7,12]
   - Conv_0_Relu_ss[7,12] = 0
 Conv_0_Relu_constraint_1[7,13]: - Conv_0_t[7,13] + Conv_0_Relu_ts[7,13]
   - Conv_0_Relu_ss[7,13] = 0
 Conv_0_Relu_constraint_1[7,14]: - Conv_0_t[7,14] + Conv_0_Relu_ts[7,14]
   - Conv_0_Relu_ss[7,14] = 0
 Conv_0_Relu_constraint_1[7,15]: - Conv_0_t[7,15] + Conv_0_Relu_ts[7,15]
   - Conv_0_Relu_ss[7,15] = 0
 Conv_0_Relu_constraint_2[0,0]: Conv_0_Relu_ts[0,0] <= 0
 Conv_0_Relu_constraint_2[0,1]: Conv_0_Relu_ts[0,1]
   - 9.188304475042969 C465 <= 0
 Conv_0_Relu_constraint_2[0,2]: Conv_0_Relu_ts[0,2] - 2.20557464659214 C466
   <= 0
 Conv_0_Relu_constraint_2[0,3]: Conv_0_Relu_ts[0,3] <= 0
 Conv_0_Relu_constraint_2[0,4]: Conv_0_Relu_ts[0,4] <= 0
 Conv_0_Relu_constraint_2[0,5]: Conv_0_Relu_ts[0,5] <= 0
 Conv_0_Relu_constraint_2[0,6]: Conv_0_Relu_ts[0,6] <= 0
 Conv_0_Relu_constraint_2[0,7]: Conv_0_Relu_ts[0,7] <= 0
 Conv_0_Relu_constraint_2[0,8]: Conv_0_Relu_ts[0,8] <= 0
 Conv_0_Relu_constraint_2[0,9]: Conv_0_Relu_ts[0,9] <= 0
 Conv_0_Relu_constraint_2[0,10]: Conv_0_Relu_ts[0,10]
   - 3.856072187423706 C474 <= 0
 Conv_0_Relu_constraint_2[0,11]: Conv_0_Relu_ts[0,11] <= 0
 Conv_0_Relu_constraint_2[0,12]: Conv_0_Relu_ts[0,12]
   - 4.69843602180481 C476 <= 0
 Conv_0_Relu_constraint_2[0,13]: Conv_0_Relu_ts[0,13] - 8.63833e-06 C477
   <= 0
 Conv_0_Relu_constraint_2[0,14]: Conv_0_Relu_ts[0,14] <= 0
 Conv_0_Relu_constraint_2[0,15]: Conv_0_Relu_ts[0,15] <= 0
 Conv_0_Relu_constraint_2[1,0]: Conv_0_Relu_ts[1,0] <= 0
 Conv_0_Relu_constraint_2[1,1]: Conv_0_Relu_ts[1,1]
   - 9.188304475042969 C481 <= 0
 Conv_0_Relu_constraint_2[1,2]: Conv_0_Relu_ts[1,2] - 2.20557464659214 C482
   <= 0
 Conv_0_Relu_constraint_2[1,3]: Conv_0_Relu_ts[1,3] <= 0
 Conv_0_Relu_constraint_2[1,4]: Conv_0_Relu_ts[1,4] <= 0
 Conv_0_Relu_constraint_2[1,5]: Conv_0_Relu_ts[1,5] <= 0
 Conv_0_Relu_constraint_2[1,6]: Conv_0_Relu_ts[1,6] <= 0
 Conv_0_Relu_constraint_2[1,7]: Conv_0_Relu_ts[1,7] <= 0
 Conv_0_Relu_constraint_2[1,8]: Conv_0_Relu_ts[1,8] <= 0
 Conv_0_Relu_constraint_2[1,9]: Conv_0_Relu_ts[1,9] <= 0
 Conv_0_Relu_constraint_2[1,10]: Conv_0_Relu_ts[1,10]
   - 3.856072187423706 C490 <= 0
 Conv_0_Relu_constraint_2[1,11]: Conv_0_Relu_ts[1,11] <= 0
 Conv_0_Relu_constraint_2[1,12]: Conv_0_Relu_ts[1,12]
   - 4.69843602180481 C492 <= 0
 Conv_0_Relu_constraint_2[1,13]: Conv_0_Relu_ts[1,13] - 8.63833e-06 C493
   <= 0
 Conv_0_Relu_constraint_2[1,14]: Conv_0_Relu_ts[1,14] <= 0
 Conv_0_Relu_constraint_2[1,15]: Conv_0_Relu_ts[1,15] <= 0
 Conv_0_Relu_constraint_2[2,0]: Conv_0_Relu_ts[2,0] <= 0
 Conv_0_Relu_constraint_2[2,1]: Conv_0_Relu_ts[2,1]
   - 9.188304475042969 C497 <= 0
 Conv_0_Relu_constraint_2[2,2]: Conv_0_Relu_ts[2,2] - 2.20557464659214 C498
   <= 0
 Conv_0_Relu_constraint_2[2,3]: Conv_0_Relu_ts[2,3] <= 0
 Conv_0_Relu_constraint_2[2,4]: Conv_0_Relu_ts[2,4] <= 0
 Conv_0_Relu_constraint_2[2,5]: Conv_0_Relu_ts[2,5] <= 0
 Conv_0_Relu_constraint_2[2,6]: Conv_0_Relu_ts[2,6] <= 0
 Conv_0_Relu_constraint_2[2,7]: Conv_0_Relu_ts[2,7] <= 0
 Conv_0_Relu_constraint_2[2,8]: Conv_0_Relu_ts[2,8] <= 0
 Conv_0_Relu_constraint_2[2,9]: Conv_0_Relu_ts[2,9] <= 0
 Conv_0_Relu_constraint_2[2,10]: Conv_0_Relu_ts[2,10]
   - 3.856072187423706 C506 <= 0
 Conv_0_Relu_constraint_2[2,11]: Conv_0_Relu_ts[2,11] <= 0
 Conv_0_Relu_constraint_2[2,12]: Conv_0_Relu_ts[2,12]
   - 4.69843602180481 C508 <= 0
 Conv_0_Relu_constraint_2[2,13]: Conv_0_Relu_ts[2,13] - 8.63833e-06 C509
   <= 0
 Conv_0_Relu_constraint_2[2,14]: Conv_0_Relu_ts[2,14] <= 0
 Conv_0_Relu_constraint_2[2,15]: Conv_0_Relu_ts[2,15] <= 0
 Conv_0_Relu_constraint_2[3,0]: Conv_0_Relu_ts[3,0] <= 0
 Conv_0_Relu_constraint_2[3,1]: Conv_0_Relu_ts[3,1]
   - 9.188304475042969 C513 <= 0
 Conv_0_Relu_constraint_2[3,2]: Conv_0_Relu_ts[3,2] - 2.20557464659214 C514
   <= 0
 Conv_0_Relu_constraint_2[3,3]: Conv_0_Relu_ts[3,3] <= 0
 Conv_0_Relu_constraint_2[3,4]: Conv_0_Relu_ts[3,4] <= 0
 Conv_0_Relu_constraint_2[3,5]: Conv_0_Relu_ts[3,5] <= 0
 Conv_0_Relu_constraint_2[3,6]: Conv_0_Relu_ts[3,6] <= 0
 Conv_0_Relu_constraint_2[3,7]: Conv_0_Relu_ts[3,7] <= 0
 Conv_0_Relu_constraint_2[3,8]: Conv_0_Relu_ts[3,8] <= 0
 Conv_0_Relu_constraint_2[3,9]: Conv_0_Relu_ts[3,9] <= 0
 Conv_0_Relu_constraint_2[3,10]: Conv_0_Relu_ts[3,10]
   - 3.856072187423706 C522 <= 0
 Conv_0_Relu_constraint_2[3,11]: Conv_0_Relu_ts[3,11] <= 0
 Conv_0_Relu_constraint_2[3,12]: Conv_0_Relu_ts[3,12]
   - 4.69843602180481 C524 <= 0
 Conv_0_Relu_constraint_2[3,13]: Conv_0_Relu_ts[3,13] - 8.63833e-06 C525
   <= 0
 Conv_0_Relu_constraint_2[3,14]: Conv_0_Relu_ts[3,14] <= 0
 Conv_0_Relu_constraint_2[3,15]: Conv_0_Relu_ts[3,15] <= 0
 Conv_0_Relu_constraint_2[4,0]: Conv_0_Relu_ts[4,0] <= 0
 Conv_0_Relu_constraint_2[4,1]: Conv_0_Relu_ts[4,1]
   - 9.188304475042969 C529 <= 0
 Conv_0_Relu_constraint_2[4,2]: Conv_0_Relu_ts[4,2] - 2.20557464659214 C530
   <= 0
 Conv_0_Relu_constraint_2[4,3]: Conv_0_Relu_ts[4,3] <= 0
 Conv_0_Relu_constraint_2[4,4]: Conv_0_Relu_ts[4,4] <= 0
 Conv_0_Relu_constraint_2[4,5]: Conv_0_Relu_ts[4,5] <= 0
 Conv_0_Relu_constraint_2[4,6]: Conv_0_Relu_ts[4,6] <= 0
 Conv_0_Relu_constraint_2[4,7]: Conv_0_Relu_ts[4,7] <= 0
 Conv_0_Relu_constraint_2[4,8]: Conv_0_Relu_ts[4,8] <= 0
 Conv_0_Relu_constraint_2[4,9]: Conv_0_Relu_ts[4,9] <= 0
 Conv_0_Relu_constraint_2[4,10]: Conv_0_Relu_ts[4,10]
   - 3.856072187423706 C538 <= 0
 Conv_0_Relu_constraint_2[4,11]: Conv_0_Relu_ts[4,11] <= 0
 Conv_0_Relu_constraint_2[4,12]: Conv_0_Relu_ts[4,12]
   - 4.69843602180481 C540 <= 0
 Conv_0_Relu_constraint_2[4,13]: Conv_0_Relu_ts[4,13] - 8.63833e-06 C541
   <= 0
 Conv_0_Relu_constraint_2[4,14]: Conv_0_Relu_ts[4,14] <= 0
 Conv_0_Relu_constraint_2[4,15]: Conv_0_Relu_ts[4,15] <= 0
 Conv_0_Relu_constraint_2[5,0]: Conv_0_Relu_ts[5,0] <= 0
 Conv_0_Relu_constraint_2[5,1]: Conv_0_Relu_ts[5,1]
   - 9.188304475042969 C545 <= 0
 Conv_0_Relu_constraint_2[5,2]: Conv_0_Relu_ts[5,2] - 2.20557464659214 C546
   <= 0
 Conv_0_Relu_constraint_2[5,3]: Conv_0_Relu_ts[5,3] <= 0
 Conv_0_Relu_constraint_2[5,4]: Conv_0_Relu_ts[5,4] <= 0
 Conv_0_Relu_constraint_2[5,5]: Conv_0_Relu_ts[5,5] <= 0
 Conv_0_Relu_constraint_2[5,6]: Conv_0_Relu_ts[5,6] <= 0
 Conv_0_Relu_constraint_2[5,7]: Conv_0_Relu_ts[5,7] <= 0
 Conv_0_Relu_constraint_2[5,8]: Conv_0_Relu_ts[5,8] <= 0
 Conv_0_Relu_constraint_2[5,9]: Conv_0_Relu_ts[5,9] <= 0
 Conv_0_Relu_constraint_2[5,10]: Conv_0_Relu_ts[5,10]
   - 3.856072187423706 C554 <= 0
 Conv_0_Relu_constraint_2[5,11]: Conv_0_Relu_ts[5,11] <= 0
 Conv_0_Relu_constraint_2[5,12]: Conv_0_Relu_ts[5,12]
   - 4.69843602180481 C556 <= 0
 Conv_0_Relu_constraint_2[5,13]: Conv_0_Relu_ts[5,13] - 8.63833e-06 C557
   <= 0
 Conv_0_Relu_constraint_2[5,14]: Conv_0_Relu_ts[5,14] <= 0
 Conv_0_Relu_constraint_2[5,15]: Conv_0_Relu_ts[5,15] <= 0
 Conv_0_Relu_constraint_2[6,0]: Conv_0_Relu_ts[6,0] <= 0
 Conv_0_Relu_constraint_2[6,1]: Conv_0_Relu_ts[6,1]
   - 9.188304475042969 C561 <= 0
 Conv_0_Relu_constraint_2[6,2]: Conv_0_Relu_ts[6,2] - 2.20557464659214 C562
   <= 0
 Conv_0_Relu_constraint_2[6,3]: Conv_0_Relu_ts[6,3] <= 0
 Conv_0_Relu_constraint_2[6,4]: Conv_0_Relu_ts[6,4] <= 0
 Conv_0_Relu_constraint_2[6,5]: Conv_0_Relu_ts[6,5] <= 0
 Conv_0_Relu_constraint_2[6,6]: Conv_0_Relu_ts[6,6] <= 0
 Conv_0_Relu_constraint_2[6,7]: Conv_0_Relu_ts[6,7] <= 0
 Conv_0_Relu_constraint_2[6,8]: Conv_0_Relu_ts[6,8] <= 0
 Conv_0_Relu_constraint_2[6,9]: Conv_0_Relu_ts[6,9] <= 0
 Conv_0_Relu_constraint_2[6,10]: Conv_0_Relu_ts[6,10]
   - 3.856072187423706 C570 <= 0
 Conv_0_Relu_constraint_2[6,11]: Conv_0_Relu_ts[6,11] <= 0
 Conv_0_Relu_constraint_2[6,12]: Conv_0_Relu_ts[6,12]
   - 4.69843602180481 C572 <= 0
 Conv_0_Relu_constraint_2[6,13]: Conv_0_Relu_ts[6,13] - 8.63833e-06 C573
   <= 0
 Conv_0_Relu_constraint_2[6,14]: Conv_0_Relu_ts[6,14] <= 0
 Conv_0_Relu_constraint_2[6,15]: Conv_0_Relu_ts[6,15] <= 0
 Conv_0_Relu_constraint_2[7,0]: Conv_0_Relu_ts[7,0] <= 0
 Conv_0_Relu_constraint_2[7,1]: Conv_0_Relu_ts[7,1]
   - 9.188304475042969 C577 <= 0
 Conv_0_Relu_constraint_2[7,2]: Conv_0_Relu_ts[7,2] - 2.20557464659214 C578
   <= 0
 Conv_0_Relu_constraint_2[7,3]: Conv_0_Relu_ts[7,3] <= 0
 Conv_0_Relu_constraint_2[7,4]: Conv_0_Relu_ts[7,4] <= 0
 Conv_0_Relu_constraint_2[7,5]: Conv_0_Relu_ts[7,5] <= 0
 Conv_0_Relu_constraint_2[7,6]: Conv_0_Relu_ts[7,6] <= 0
 Conv_0_Relu_constraint_2[7,7]: Conv_0_Relu_ts[7,7] <= 0
 Conv_0_Relu_constraint_2[7,8]: Conv_0_Relu_ts[7,8] <= 0
 Conv_0_Relu_constraint_2[7,9]: Conv_0_Relu_ts[7,9] <= 0
 Conv_0_Relu_constraint_2[7,10]: Conv_0_Relu_ts[7,10]
   - 3.856072187423706 C586 <= 0
 Conv_0_Relu_constraint_2[7,11]: Conv_0_Relu_ts[7,11] <= 0
 Conv_0_Relu_constraint_2[7,12]: Conv_0_Relu_ts[7,12]
   - 4.69843602180481 C588 <= 0
 Conv_0_Relu_constraint_2[7,13]: Conv_0_Relu_ts[7,13] - 8.63833e-06 C589
   <= 0
 Conv_0_Relu_constraint_2[7,14]: Conv_0_Relu_ts[7,14] <= 0
 Conv_0_Relu_constraint_2[7,15]: Conv_0_Relu_ts[7,15] <= 0
 Conv_0_Relu_constraint_3[0,0]: Conv_0_Relu_ss[0,0]
   + 0.00345964082953287 C464 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[0,1]: Conv_0_Relu_ss[0,1]
   + 9.411394545342773 C465 <= 9.411394545342773
 Conv_0_Relu_constraint_3[0,2]: Conv_0_Relu_ss[0,2]
   + 3.183451279997826 C466 <= 3.183451279997826
 Conv_0_Relu_constraint_3[0,3]: Conv_0_Relu_ss[0,3] + 4.99524e-07 C467
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[0,4]: Conv_0_Relu_ss[0,4]
   + 0.00330359794190437 C468 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[0,5]: Conv_0_Relu_ss[0,5]
   + 0.00111407172748819 C469 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[0,6]: Conv_0_Relu_ss[0,6]
   + 0.6694869505063252 C470 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[0,7]: Conv_0_Relu_ss[0,7]
   + 0.00719343813517304 C471 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[0,8]: Conv_0_Relu_ss[0,8] + 1.23637e-06 C472
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[0,9]: Conv_0_Relu_ss[0,9] + 3.06981e-09 C473
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[0,10]: Conv_0_Relu_ss[0,10]
   + 5.203926563262939 C474 <= 5.203926563262939
 Conv_0_Relu_constraint_3[0,11]: Conv_0_Relu_ss[0,11]
   + 0.00222280243337676 C475 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[0,12]: Conv_0_Relu_ss[0,12]
   + 1.310906410217285 C476 <= 1.310906410217285
 Conv_0_Relu_constraint_3[0,13]: Conv_0_Relu_ss[0,13] + 8.71338e-05 C477
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[0,14]: Conv_0_Relu_ss[0,14] <= 0
 Conv_0_Relu_constraint_3[0,15]: Conv_0_Relu_ss[0,15] <= 2.94497e-41
 Conv_0_Relu_constraint_3[1,0]: Conv_0_Relu_ss[1,0]
   + 0.00345964082953287 C480 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[1,1]: Conv_0_Relu_ss[1,1]
   + 9.411394545342773 C481 <= 9.411394545342773
 Conv_0_Relu_constraint_3[1,2]: Conv_0_Relu_ss[1,2]
   + 3.183451279997826 C482 <= 3.183451279997826
 Conv_0_Relu_constraint_3[1,3]: Conv_0_Relu_ss[1,3] + 4.99524e-07 C483
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[1,4]: Conv_0_Relu_ss[1,4]
   + 0.00330359794190437 C484 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[1,5]: Conv_0_Relu_ss[1,5]
   + 0.00111407172748819 C485 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[1,6]: Conv_0_Relu_ss[1,6]
   + 0.6694869505063252 C486 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[1,7]: Conv_0_Relu_ss[1,7]
   + 0.00719343813517304 C487 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[1,8]: Conv_0_Relu_ss[1,8] + 1.23637e-06 C488
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[1,9]: Conv_0_Relu_ss[1,9] + 3.06981e-09 C489
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[1,10]: Conv_0_Relu_ss[1,10]
   + 5.203926563262939 C490 <= 5.203926563262939
 Conv_0_Relu_constraint_3[1,11]: Conv_0_Relu_ss[1,11]
   + 0.00222280243337676 C491 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[1,12]: Conv_0_Relu_ss[1,12]
   + 1.310906410217285 C492 <= 1.310906410217285
 Conv_0_Relu_constraint_3[1,13]: Conv_0_Relu_ss[1,13] + 8.71338e-05 C493
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[1,14]: Conv_0_Relu_ss[1,14] <= 0
 Conv_0_Relu_constraint_3[1,15]: Conv_0_Relu_ss[1,15] <= 2.94497e-41
 Conv_0_Relu_constraint_3[2,0]: Conv_0_Relu_ss[2,0]
   + 0.00345964082953287 C496 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[2,1]: Conv_0_Relu_ss[2,1]
   + 9.411394545342773 C497 <= 9.411394545342773
 Conv_0_Relu_constraint_3[2,2]: Conv_0_Relu_ss[2,2]
   + 3.183451279997826 C498 <= 3.183451279997826
 Conv_0_Relu_constraint_3[2,3]: Conv_0_Relu_ss[2,3] + 4.99524e-07 C499
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[2,4]: Conv_0_Relu_ss[2,4]
   + 0.00330359794190437 C500 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[2,5]: Conv_0_Relu_ss[2,5]
   + 0.00111407172748819 C501 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[2,6]: Conv_0_Relu_ss[2,6]
   + 0.6694869505063252 C502 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[2,7]: Conv_0_Relu_ss[2,7]
   + 0.00719343813517304 C503 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[2,8]: Conv_0_Relu_ss[2,8] + 1.23637e-06 C504
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[2,9]: Conv_0_Relu_ss[2,9] + 3.06981e-09 C505
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[2,10]: Conv_0_Relu_ss[2,10]
   + 5.203926563262939 C506 <= 5.203926563262939
 Conv_0_Relu_constraint_3[2,11]: Conv_0_Relu_ss[2,11]
   + 0.00222280243337676 C507 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[2,12]: Conv_0_Relu_ss[2,12]
   + 1.310906410217285 C508 <= 1.310906410217285
 Conv_0_Relu_constraint_3[2,13]: Conv_0_Relu_ss[2,13] + 8.71338e-05 C509
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[2,14]: Conv_0_Relu_ss[2,14] <= 0
 Conv_0_Relu_constraint_3[2,15]: Conv_0_Relu_ss[2,15] <= 2.94497e-41
 Conv_0_Relu_constraint_3[3,0]: Conv_0_Relu_ss[3,0]
   + 0.00345964082953287 C512 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[3,1]: Conv_0_Relu_ss[3,1]
   + 9.411394545342773 C513 <= 9.411394545342773
 Conv_0_Relu_constraint_3[3,2]: Conv_0_Relu_ss[3,2]
   + 3.183451279997826 C514 <= 3.183451279997826
 Conv_0_Relu_constraint_3[3,3]: Conv_0_Relu_ss[3,3] + 4.99524e-07 C515
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[3,4]: Conv_0_Relu_ss[3,4]
   + 0.00330359794190437 C516 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[3,5]: Conv_0_Relu_ss[3,5]
   + 0.00111407172748819 C517 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[3,6]: Conv_0_Relu_ss[3,6]
   + 0.6694869505063252 C518 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[3,7]: Conv_0_Relu_ss[3,7]
   + 0.00719343813517304 C519 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[3,8]: Conv_0_Relu_ss[3,8] + 1.23637e-06 C520
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[3,9]: Conv_0_Relu_ss[3,9] + 3.06981e-09 C521
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[3,10]: Conv_0_Relu_ss[3,10]
   + 5.203926563262939 C522 <= 5.203926563262939
 Conv_0_Relu_constraint_3[3,11]: Conv_0_Relu_ss[3,11]
   + 0.00222280243337676 C523 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[3,12]: Conv_0_Relu_ss[3,12]
   + 1.310906410217285 C524 <= 1.310906410217285
 Conv_0_Relu_constraint_3[3,13]: Conv_0_Relu_ss[3,13] + 8.71338e-05 C525
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[3,14]: Conv_0_Relu_ss[3,14] <= 0
 Conv_0_Relu_constraint_3[3,15]: Conv_0_Relu_ss[3,15] <= 2.94497e-41
 Conv_0_Relu_constraint_3[4,0]: Conv_0_Relu_ss[4,0]
   + 0.00345964082953287 C528 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[4,1]: Conv_0_Relu_ss[4,1]
   + 9.411394545342773 C529 <= 9.411394545342773
 Conv_0_Relu_constraint_3[4,2]: Conv_0_Relu_ss[4,2]
   + 3.183451279997826 C530 <= 3.183451279997826
 Conv_0_Relu_constraint_3[4,3]: Conv_0_Relu_ss[4,3] + 4.99524e-07 C531
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[4,4]: Conv_0_Relu_ss[4,4]
   + 0.00330359794190437 C532 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[4,5]: Conv_0_Relu_ss[4,5]
   + 0.00111407172748819 C533 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[4,6]: Conv_0_Relu_ss[4,6]
   + 0.6694869505063252 C534 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[4,7]: Conv_0_Relu_ss[4,7]
   + 0.00719343813517304 C535 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[4,8]: Conv_0_Relu_ss[4,8] + 1.23637e-06 C536
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[4,9]: Conv_0_Relu_ss[4,9] + 3.06981e-09 C537
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[4,10]: Conv_0_Relu_ss[4,10]
   + 5.203926563262939 C538 <= 5.203926563262939
 Conv_0_Relu_constraint_3[4,11]: Conv_0_Relu_ss[4,11]
   + 0.00222280243337676 C539 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[4,12]: Conv_0_Relu_ss[4,12]
   + 1.310906410217285 C540 <= 1.310906410217285
 Conv_0_Relu_constraint_3[4,13]: Conv_0_Relu_ss[4,13] + 8.71338e-05 C541
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[4,14]: Conv_0_Relu_ss[4,14] <= 0
 Conv_0_Relu_constraint_3[4,15]: Conv_0_Relu_ss[4,15] <= 2.94497e-41
 Conv_0_Relu_constraint_3[5,0]: Conv_0_Relu_ss[5,0]
   + 0.00345964082953287 C544 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[5,1]: Conv_0_Relu_ss[5,1]
   + 9.411394545342773 C545 <= 9.411394545342773
 Conv_0_Relu_constraint_3[5,2]: Conv_0_Relu_ss[5,2]
   + 3.183451279997826 C546 <= 3.183451279997826
 Conv_0_Relu_constraint_3[5,3]: Conv_0_Relu_ss[5,3] + 4.99524e-07 C547
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[5,4]: Conv_0_Relu_ss[5,4]
   + 0.00330359794190437 C548 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[5,5]: Conv_0_Relu_ss[5,5]
   + 0.00111407172748819 C549 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[5,6]: Conv_0_Relu_ss[5,6]
   + 0.6694869505063252 C550 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[5,7]: Conv_0_Relu_ss[5,7]
   + 0.00719343813517304 C551 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[5,8]: Conv_0_Relu_ss[5,8] + 1.23637e-06 C552
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[5,9]: Conv_0_Relu_ss[5,9] + 3.06981e-09 C553
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[5,10]: Conv_0_Relu_ss[5,10]
   + 5.203926563262939 C554 <= 5.203926563262939
 Conv_0_Relu_constraint_3[5,11]: Conv_0_Relu_ss[5,11]
   + 0.00222280243337676 C555 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[5,12]: Conv_0_Relu_ss[5,12]
   + 1.310906410217285 C556 <= 1.310906410217285
 Conv_0_Relu_constraint_3[5,13]: Conv_0_Relu_ss[5,13] + 8.71338e-05 C557
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[5,14]: Conv_0_Relu_ss[5,14] <= 0
 Conv_0_Relu_constraint_3[5,15]: Conv_0_Relu_ss[5,15] <= 2.94497e-41
 Conv_0_Relu_constraint_3[6,0]: Conv_0_Relu_ss[6,0]
   + 0.00345964082953287 C560 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[6,1]: Conv_0_Relu_ss[6,1]
   + 9.411394545342773 C561 <= 9.411394545342773
 Conv_0_Relu_constraint_3[6,2]: Conv_0_Relu_ss[6,2]
   + 3.183451279997826 C562 <= 3.183451279997826
 Conv_0_Relu_constraint_3[6,3]: Conv_0_Relu_ss[6,3] + 4.99524e-07 C563
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[6,4]: Conv_0_Relu_ss[6,4]
   + 0.00330359794190437 C564 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[6,5]: Conv_0_Relu_ss[6,5]
   + 0.00111407172748819 C565 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[6,6]: Conv_0_Relu_ss[6,6]
   + 0.6694869505063252 C566 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[6,7]: Conv_0_Relu_ss[6,7]
   + 0.00719343813517304 C567 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[6,8]: Conv_0_Relu_ss[6,8] + 1.23637e-06 C568
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[6,9]: Conv_0_Relu_ss[6,9] + 3.06981e-09 C569
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[6,10]: Conv_0_Relu_ss[6,10]
   + 5.203926563262939 C570 <= 5.203926563262939
 Conv_0_Relu_constraint_3[6,11]: Conv_0_Relu_ss[6,11]
   + 0.00222280243337676 C571 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[6,12]: Conv_0_Relu_ss[6,12]
   + 1.310906410217285 C572 <= 1.310906410217285
 Conv_0_Relu_constraint_3[6,13]: Conv_0_Relu_ss[6,13] + 8.71338e-05 C573
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[6,14]: Conv_0_Relu_ss[6,14] <= 0
 Conv_0_Relu_constraint_3[6,15]: Conv_0_Relu_ss[6,15] <= 2.94497e-41
 Conv_0_Relu_constraint_3[7,0]: Conv_0_Relu_ss[7,0]
   + 0.00345964082953287 C576 <= 0.00345964082953287
 Conv_0_Relu_constraint_3[7,1]: Conv_0_Relu_ss[7,1]
   + 9.411394545342773 C577 <= 9.411394545342773
 Conv_0_Relu_constraint_3[7,2]: Conv_0_Relu_ss[7,2]
   + 3.183451279997826 C578 <= 3.183451279997826
 Conv_0_Relu_constraint_3[7,3]: Conv_0_Relu_ss[7,3] + 4.99524e-07 C579
   <= 4.99524e-07
 Conv_0_Relu_constraint_3[7,4]: Conv_0_Relu_ss[7,4]
   + 0.00330359794190437 C580 <= 0.00330359794190437
 Conv_0_Relu_constraint_3[7,5]: Conv_0_Relu_ss[7,5]
   + 0.00111407172748819 C581 <= 0.00111407172748819
 Conv_0_Relu_constraint_3[7,6]: Conv_0_Relu_ss[7,6]
   + 0.6694869505063252 C582 <= 0.6694869505063252
 Conv_0_Relu_constraint_3[7,7]: Conv_0_Relu_ss[7,7]
   + 0.00719343813517304 C583 <= 0.00719343813517304
 Conv_0_Relu_constraint_3[7,8]: Conv_0_Relu_ss[7,8] + 1.23637e-06 C584
   <= 1.23637e-06
 Conv_0_Relu_constraint_3[7,9]: Conv_0_Relu_ss[7,9] + 3.06981e-09 C585
   <= 3.06981e-09
 Conv_0_Relu_constraint_3[7,10]: Conv_0_Relu_ss[7,10]
   + 5.203926563262939 C586 <= 5.203926563262939
 Conv_0_Relu_constraint_3[7,11]: Conv_0_Relu_ss[7,11]
   + 0.00222280243337676 C587 <= 0.00222280243337676
 Conv_0_Relu_constraint_3[7,12]: Conv_0_Relu_ss[7,12]
   + 1.310906410217285 C588 <= 1.310906410217285
 Conv_0_Relu_constraint_3[7,13]: Conv_0_Relu_ss[7,13] + 8.71338e-05 C589
   <= 8.71338e-05
 Conv_0_Relu_constraint_3[7,14]: Conv_0_Relu_ss[7,14] <= 0
 Conv_0_Relu_constraint_3[7,15]: Conv_0_Relu_ss[7,15] <= 2.94497e-41
 Conv_1_output_constraint[0,0]: 1.69823e-06 Conv_0_Relu_ts[0,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[0,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[0,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[0,12]
   + Conv_1_t[0,0] = -1.7536e-08
 Conv_1_output_constraint[0,1]: 6.64218e-05 Conv_0_Relu_ts[0,0]
   - 1.304573059082031 Conv_0_Relu_ts[0,1]
   - 0.00808643829077482 Conv_0_Relu_ts[0,2]
   + 1.10233e-12 Conv_0_Relu_ts[0,4] - 6.46578e-08 Conv_0_Relu_ts[0,6]
   + 3.59547e-13 Conv_0_Relu_ts[0,7] + 2.26512e-09 Conv_0_Relu_ts[0,8]
   - 1.48387e-10 Conv_0_Relu_ts[0,9]
   - 0.0375515259802341 Conv_0_Relu_ts[0,10]
   - 0.1779745519161224 Conv_0_Relu_ts[0,12]
   - 1.40572e-06 Conv_0_Relu_ts[0,13]
   + 4.89468e-05 Conv_1_aggregated_features[0,0]
   - 0.00799453444778919 Conv_1_aggregated_features[0,1]
   - 0.0869906693696976 Conv_1_aggregated_features[0,2]
   - 3.80301e-08 Conv_1_aggregated_features[0,6]
   + 2.46415e-09 Conv_1_aggregated_features[0,8]
   - 1.62219e-10 Conv_1_aggregated_features[0,9]
   - 0.2601335346698761 Conv_1_aggregated_features[0,10]
   + 0.4667477905750275 Conv_1_aggregated_features[0,12]
   - 1.71701e-06 Conv_1_aggregated_features[0,13] + Conv_1_t[0,1]
   = 0.3359648287296295
 Conv_1_output_constraint[0,2]: - 2.68276e-05 Conv_0_Relu_ts[0,0]
   + 7.4485e-11 Conv_0_Relu_ts[0,1] + 3.46068e-11 Conv_0_Relu_ts[0,2]
   - 5.94579e-10 Conv_0_Relu_ts[0,4] + 1.49975e-09 Conv_0_Relu_ts[0,5]
   + 4.00629e-09 Conv_0_Relu_ts[0,6] - 0.000385428 Conv_0_Relu_ts[0,8]
   - 9.65733e-05 Conv_0_Relu_ts[0,9] + 3.04566e-10 Conv_0_Relu_ts[0,10]
   + 2.67322e-05 Conv_0_Relu_ts[0,11] + 5.13886e-09 Conv_0_Relu_ts[0,12]
   - 6.79858e-13 Conv_0_Relu_ts[0,13]
   + 2.85913e-06 Conv_1_aggregated_features[0,0]
   + 3.02975e-09 Conv_1_aggregated_features[0,1]
   + 6.8068e-11 Conv_1_aggregated_features[0,2]
   + 2.87298e-07 Conv_1_aggregated_features[0,4]
   + 1.28094e-07 Conv_1_aggregated_features[0,5]
   + 3.03159e-11 Conv_1_aggregated_features[0,6]
   + 2.96378e-05 Conv_1_aggregated_features[0,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[0,9]
   + 7.60786e-11 Conv_1_aggregated_features[0,10]
   + 9.73847e-08 Conv_1_aggregated_features[0,11]
   + 7.54748e-09 Conv_1_aggregated_features[0,12]
   + 1.58189e-09 Conv_1_aggregated_features[0,13] + Conv_1_t[0,2]
   = -1.53413e-08
 Conv_1_output_constraint[0,3]: 9.55069e-08 Conv_0_Relu_ts[0,3]
   - 4.06432e-09 Conv_0_Relu_ts[0,5] - 2.23188e-09 Conv_0_Relu_ts[0,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[0,12]
   + 4.18428e-06 Conv_0_Relu_ts[0,13]
   + 7.14864e-07 Conv_1_aggregated_features[0,3]
   - 9.90425e-11 Conv_1_aggregated_features[0,5]
   + 2.85992e-07 Conv_1_aggregated_features[0,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[0,12]
   - 7.41084e-08 Conv_1_aggregated_features[0,13] + Conv_1_t[0,3]
   = -3.15407e-06
 Conv_1_output_constraint[0,4]: 2.97121e-10 Conv_0_Relu_ts[0,0]
   + 9.07965e-08 Conv_0_Relu_ts[0,1] + 1.13683e-08 Conv_0_Relu_ts[0,2]
   + 7.68037e-08 Conv_0_Relu_ts[0,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[0,5]
   - 5.39198e-07 Conv_0_Relu_ts[0,6] - 6.18604e-12 Conv_0_Relu_ts[0,8]
   + 3.00799e-08 Conv_0_Relu_ts[0,10] - 1.66347e-05 Conv_0_Relu_ts[0,11]
   + 2.56472e-07 Conv_0_Relu_ts[0,12] - 3.27559e-05 Conv_0_Relu_ts[0,13]
   - 6.3717e-09 Conv_1_aggregated_features[0,0]
   + 3.53659e-08 Conv_1_aggregated_features[0,1]
   + 7.41964e-09 Conv_1_aggregated_features[0,2]
   + 0.00146958860568702 Conv_1_aggregated_features[0,4]
   + 1.41886e-06 Conv_1_aggregated_features[0,5]
   - 1.75898e-05 Conv_1_aggregated_features[0,6]
   + 0.0017772619612515 Conv_1_aggregated_features[0,8]
   + 7.64123e-10 Conv_1_aggregated_features[0,9]
   + 2.4839e-08 Conv_1_aggregated_features[0,10]
   - 2.68593e-07 Conv_1_aggregated_features[0,11]
   + 2.04143e-07 Conv_1_aggregated_features[0,12]
   + 4.75765e-08 Conv_1_aggregated_features[0,13] + Conv_1_t[0,4]
   = -8.58295e-08
 Conv_1_output_constraint[0,5]: - 1.12261e-07 Conv_0_Relu_ts[0,0]
   - 6.16837e-11 Conv_0_Relu_ts[0,1] - 1.43211e-10 Conv_0_Relu_ts[0,2]
   + 9.76965e-10 Conv_0_Relu_ts[0,3] - 8.36344e-05 Conv_0_Relu_ts[0,4]
   + 2.12422e-08 Conv_0_Relu_ts[0,5] - 2.04737e-07 Conv_0_Relu_ts[0,6]
   - 6.84967e-06 Conv_0_Relu_ts[0,8] + 2.60175e-05 Conv_0_Relu_ts[0,9]
   + 1.5048e-10 Conv_0_Relu_ts[0,10] - 2.26739e-07 Conv_0_Relu_ts[0,11]
   + 2.04031e-09 Conv_0_Relu_ts[0,12] + 3.64928e-12 Conv_0_Relu_ts[0,13]
   - 4.59272e-08 Conv_0_Relu_ts[0,15]
   + 8.09056e-08 Conv_1_aggregated_features[0,0]
   + 1.22547e-10 Conv_1_aggregated_features[0,1]
   + 1.29284e-10 Conv_1_aggregated_features[0,2]
   + 5.76885e-11 Conv_1_aggregated_features[0,3]
   + 5.76747e-08 Conv_1_aggregated_features[0,4]
   + 1.49451e-09 Conv_1_aggregated_features[0,5]
   + 9.85149e-10 Conv_1_aggregated_features[0,8]
   + 1.88005e-11 Conv_1_aggregated_features[0,9]
   + 3.09409e-11 Conv_1_aggregated_features[0,10]
   + 1.45468e-07 Conv_1_aggregated_features[0,11]
   + 2.76313e-09 Conv_1_aggregated_features[0,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[0,13]
   + Conv_1_t[0,5] = -4.66972e-09
 Conv_1_output_constraint[0,6]: 1.68259e-08 Conv_0_Relu_ts[0,0]
   + 2.66449e-06 Conv_0_Relu_ts[0,1] - 3.07804e-06 Conv_0_Relu_ts[0,2]
   - 1.56854e-10 Conv_0_Relu_ts[0,3] + 1.61577e-06 Conv_0_Relu_ts[0,4]
   + 9.63312e-13 Conv_0_Relu_ts[0,5] + 8.94799e-11 Conv_0_Relu_ts[0,6]
   + 0.00156858959235251 Conv_0_Relu_ts[0,8]
   + 3.71009e-11 Conv_0_Relu_ts[0,9] - 1.18851e-06 Conv_0_Relu_ts[0,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[0,11]
   + 2.08067e-05 Conv_0_Relu_ts[0,12] - 1.93794e-06 Conv_0_Relu_ts[0,13]
   + 2.00933e-08 Conv_1_aggregated_features[0,0]
   + 1.84936e-05 Conv_1_aggregated_features[0,1]
   - 2.40274e-06 Conv_1_aggregated_features[0,2]
   + 4.16152e-12 Conv_1_aggregated_features[0,3]
   - 9.5418e-06 Conv_1_aggregated_features[0,4]
   + 2.61288e-12 Conv_1_aggregated_features[0,5]
   - 5.0478e-05 Conv_1_aggregated_features[0,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[0,8]
   - 4.57774e-07 Conv_1_aggregated_features[0,9]
   - 7.00605e-07 Conv_1_aggregated_features[0,10]
   + 2.2587e-10 Conv_1_aggregated_features[0,11]
   + 4.15987e-05 Conv_1_aggregated_features[0,12]
   - 1.80554e-09 Conv_1_aggregated_features[0,13] + Conv_1_t[0,6]
   = -9.69776e-05
 Conv_1_output_constraint[0,7]: - 1.58234e-12 Conv_0_Relu_ts[0,0]
   + 6.43182e-10 Conv_0_Relu_ts[0,1] - 1.24733e-10 Conv_0_Relu_ts[0,2]
   + 7.71679e-06 Conv_0_Relu_ts[0,4] + 2.78803e-06 Conv_0_Relu_ts[0,6]
   + 2.03373e-06 Conv_0_Relu_ts[0,8] - 2.93243e-11 Conv_0_Relu_ts[0,9]
   - 2.23517e-10 Conv_0_Relu_ts[0,10] + 2.01832e-09 Conv_0_Relu_ts[0,12]
   + 5.32355e-07 Conv_1_aggregated_features[0,0]
   + 8.4797e-09 Conv_1_aggregated_features[0,1]
   - 1.50954e-10 Conv_1_aggregated_features[0,2]
   + 1.6387e-09 Conv_1_aggregated_features[0,4]
   + 2.45145e-06 Conv_1_aggregated_features[0,6]
   + 1.09862e-10 Conv_1_aggregated_features[0,7]
   - 1.68113e-05 Conv_1_aggregated_features[0,8]
   + 1.4449e-10 Conv_1_aggregated_features[0,9]
   - 4.62632e-10 Conv_1_aggregated_features[0,10]
   + 1.80231e-09 Conv_1_aggregated_features[0,12] + Conv_1_t[0,7]
   = -6.52043e-09
 Conv_1_output_constraint[0,8]: 2.0398e-13 Conv_1_aggregated_features[0,12]
   + Conv_1_t[0,8] = 2.40312e-18
 Conv_1_output_constraint[0,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[0,0]
   + 0.4666179716587067 Conv_0_Relu_ts[0,1]
   - 0.212794229388237 Conv_0_Relu_ts[0,2]
   + 1.31709e-13 Conv_0_Relu_ts[0,3] + 6.41139e-10 Conv_0_Relu_ts[0,4]
   - 1.36516e-10 Conv_0_Relu_ts[0,5]
   + 0.00123115978203714 Conv_0_Relu_ts[0,6]
   - 3.8707e-07 Conv_0_Relu_ts[0,7] - 9.99069e-10 Conv_0_Relu_ts[0,8]
   - 3.90229e-11 Conv_0_Relu_ts[0,9]
   - 0.3983311057090759 Conv_0_Relu_ts[0,10]
   + 1.4696e-09 Conv_0_Relu_ts[0,11]
   + 0.135991245508194 Conv_0_Relu_ts[0,12]
   - 5.03958e-07 Conv_0_Relu_ts[0,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[0,0]
   + 0.0555339716374874 Conv_1_aggregated_features[0,1]
   - 0.1152108833193779 Conv_1_aggregated_features[0,2]
   + 1.01018e-09 Conv_1_aggregated_features[0,4]
   - 2.65221e-10 Conv_1_aggregated_features[0,5]
   + 0.00134999 Conv_1_aggregated_features[0,6]
   - 7.7429e-07 Conv_1_aggregated_features[0,7]
   - 9.0928e-10 Conv_1_aggregated_features[0,8]
   - 4.02666e-11 Conv_1_aggregated_features[0,9]
   - 0.1311033517122269 Conv_1_aggregated_features[0,10]
   + 5.72141e-09 Conv_1_aggregated_features[0,11]
   - 1.062317252159119 Conv_1_aggregated_features[0,12]
   - 3.93908e-07 Conv_1_aggregated_features[0,13] + Conv_1_t[0,9]
   = 0.4778489172458649
 Conv_1_output_constraint[0,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[0,0]
   + 3.12754e-07 Conv_0_Relu_ts[0,1] + 1.78384e-08 Conv_0_Relu_ts[0,2]
   - 8.1105e-12 Conv_0_Relu_ts[0,3] - 1.21368e-06 Conv_0_Relu_ts[0,4]
   - 7.58218e-10 Conv_0_Relu_ts[0,5] + 6.37659e-05 Conv_0_Relu_ts[0,7]
   + 5.06994e-09 Conv_0_Relu_ts[0,8] + 4.38429e-08 Conv_0_Relu_ts[0,10]
   + 1.00038e-06 Conv_0_Relu_ts[0,12] - 1.00897e-07 Conv_0_Relu_ts[0,13]
   - 7.72898e-10 Conv_1_aggregated_features[0,0]
   + 1.93344e-07 Conv_1_aggregated_features[0,1]
   + 2.31957e-08 Conv_1_aggregated_features[0,2]
   - 5.28535e-07 Conv_1_aggregated_features[0,3]
   - 8.50617e-06 Conv_1_aggregated_features[0,4]
   - 1.86843e-09 Conv_1_aggregated_features[0,5]
   - 4.42334e-07 Conv_1_aggregated_features[0,8]
   - 1.66725e-07 Conv_1_aggregated_features[0,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[0,11]
   + 9.07032e-07 Conv_1_aggregated_features[0,12]
   + 1.19635e-11 Conv_1_aggregated_features[0,13] + Conv_1_t[0,10]
   = -7.05563e-07
 Conv_1_output_constraint[0,11]: 1.34578e-06 Conv_0_Relu_ts[0,0]
   + 2.25868e-05 Conv_0_Relu_ts[0,1] + 1.36528e-05 Conv_0_Relu_ts[0,2]
   + 6.61139e-05 Conv_0_Relu_ts[0,4] + 1.97151e-05 Conv_0_Relu_ts[0,5]
   - 2.52237e-09 Conv_0_Relu_ts[0,8] - 2.08745e-05 Conv_0_Relu_ts[0,9]
   + 4.27105e-05 Conv_0_Relu_ts[0,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[0,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[0,12]
   + 9.27492e-09 Conv_0_Relu_ts[0,13] - 1.59817e-08 Conv_0_Relu_ts[0,15]
   - 7.80059e-05 Conv_1_aggregated_features[0,0]
   + 3.38276e-05 Conv_1_aggregated_features[0,1]
   + 4.73819e-06 Conv_1_aggregated_features[0,2]
   - 0.00209861318580806 Conv_1_aggregated_features[0,4]
   - 4.14211e-05 Conv_1_aggregated_features[0,5]
   - 1.65027e-11 Conv_1_aggregated_features[0,8]
   + 2.09826e-05 Conv_1_aggregated_features[0,10]
   + 3.24145e-10 Conv_1_aggregated_features[0,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[0,12]
   - 6.08701e-11 Conv_1_aggregated_features[0,13] + Conv_1_t[0,11]
   = 3.07161e-05
 Conv_1_output_constraint[0,12]: 7.60796e-09 Conv_0_Relu_ts[0,1]
   + 2.00773e-10 Conv_0_Relu_ts[0,2] + 3.86788e-10 Conv_0_Relu_ts[0,3]
   + 3.1184e-11 Conv_0_Relu_ts[0,8] + 1.83806e-10 Conv_0_Relu_ts[0,10]
   + 2.13717e-08 Conv_0_Relu_ts[0,12] - 6.30213e-12 Conv_0_Relu_ts[0,13]
   + 2.90536e-09 Conv_1_aggregated_features[0,1]
   - 3.16398e-10 Conv_1_aggregated_features[0,2]
   - 2.02197e-09 Conv_1_aggregated_features[0,3]
   - 1.19877e-11 Conv_1_aggregated_features[0,8]
   - 5.09457e-10 Conv_1_aggregated_features[0,10]
   + 2.27248e-08 Conv_1_aggregated_features[0,12]
   - 1.24645e-13 Conv_1_aggregated_features[0,13] + Conv_1_t[0,12]
   = -3.75308e-09
 Conv_1_output_constraint[0,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[0,0]
   - 0.1570860594511032 Conv_0_Relu_ts[0,1]
   + 0.3520863950252533 Conv_0_Relu_ts[0,2]
   - 1.29036e-08 Conv_0_Relu_ts[0,4] + 3.83205e-07 Conv_0_Relu_ts[0,5]
   - 0.00537013821303844 Conv_0_Relu_ts[0,6]
   - 8.97018e-08 Conv_0_Relu_ts[0,7] + 1.712e-10 Conv_0_Relu_ts[0,8]
   + 1.81878e-10 Conv_0_Relu_ts[0,9]
   + 0.4303593337535858 Conv_0_Relu_ts[0,10]
   - 1.61554e-08 Conv_0_Relu_ts[0,11]
   - 0.9454745650291443 Conv_0_Relu_ts[0,12]
   + 4.84941e-06 Conv_0_Relu_ts[0,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[0,0]
   - 0.7010135054588318 Conv_1_aggregated_features[0,1]
   + 0.0964145809412003 Conv_1_aggregated_features[0,2]
   - 3.74041e-08 Conv_1_aggregated_features[0,4]
   + 5.47003e-07 Conv_1_aggregated_features[0,5]
   - 0.00514170713722706 Conv_1_aggregated_features[0,6]
   - 9.53448e-08 Conv_1_aggregated_features[0,7]
   - 2.07781e-10 Conv_1_aggregated_features[0,8]
   + 1.66638e-10 Conv_1_aggregated_features[0,9]
   + 0.1311885118484497 Conv_1_aggregated_features[0,10]
   - 9.23591e-08 Conv_1_aggregated_features[0,11]
   - 0.1851733475923538 Conv_1_aggregated_features[0,12]
   + 5.0438e-06 Conv_1_aggregated_features[0,13] + Conv_1_t[0,13]
   = -1.276697397232056
 Conv_1_output_constraint[0,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[0,0]
   + 0.00206982018426061 Conv_0_Relu_ts[0,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[0,2]
   - 9.77907e-13 Conv_0_Relu_ts[0,3] - 3.82624e-07 Conv_0_Relu_ts[0,4]
   + 1.19511e-07 Conv_0_Relu_ts[0,5] - 2.54839e-11 Conv_0_Relu_ts[0,6]
   + 4.99102e-07 Conv_0_Relu_ts[0,8] + 2.64257e-11 Conv_0_Relu_ts[0,9]
   - 0.00118294754065573 Conv_0_Relu_ts[0,10]
   + 4.8915e-08 Conv_0_Relu_ts[0,11]
   + 0.00699353078380227 Conv_0_Relu_ts[0,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[0,13]
   + 1.51033e-06 Conv_1_aggregated_features[0,0]
   + 0.00578453531488776 Conv_1_aggregated_features[0,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[0,2]
   + 5.14278e-09 Conv_1_aggregated_features[0,4]
   + 3.41007e-10 Conv_1_aggregated_features[0,5]
   + 1.47387e-08 Conv_1_aggregated_features[0,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[0,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[0,10]
   + 1.3873e-07 Conv_1_aggregated_features[0,11]
   + 0.0143724931403995 Conv_1_aggregated_features[0,12]
   - 6.50481e-05 Conv_1_aggregated_features[0,13] + Conv_1_t[0,14]
   = -0.00573839386925101
 Conv_1_output_constraint[0,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[0,0]
   + 3.59972e-09 Conv_0_Relu_ts[0,1] - 2.63862e-12 Conv_0_Relu_ts[0,2]
   - 0.00124379759654403 Conv_0_Relu_ts[0,4]
   + 1.90933e-08 Conv_0_Relu_ts[0,5] - 4.63531e-13 Conv_0_Relu_ts[0,6]
   - 1.54557e-05 Conv_0_Relu_ts[0,7] - 1.60225e-10 Conv_0_Relu_ts[0,8]
   + 2.49673e-06 Conv_0_Relu_ts[0,9] + 7.17346e-12 Conv_0_Relu_ts[0,10]
   - 3.53936e-11 Conv_0_Relu_ts[0,11] + 1.06168e-09 Conv_0_Relu_ts[0,12]
   - 4.87679e-07 Conv_0_Relu_ts[0,13] + 3.14464e-11 Conv_0_Relu_ts[0,15]
   - 0.000458086 Conv_1_aggregated_features[0,0]
   + 7.50942e-09 Conv_1_aggregated_features[0,1]
   - 3.13868e-12 Conv_1_aggregated_features[0,2]
   - 5.10815e-06 Conv_1_aggregated_features[0,4]
   - 1.93728e-06 Conv_1_aggregated_features[0,5]
   - 9.15832e-12 Conv_1_aggregated_features[0,6]
   + 5.1922e-09 Conv_1_aggregated_features[0,7]
   + 2.54628e-11 Conv_1_aggregated_features[0,8]
   - 1.10428e-11 Conv_1_aggregated_features[0,10]
   + 1.00597e-09 Conv_1_aggregated_features[0,12]
   + 7.86155e-08 Conv_1_aggregated_features[0,13] + Conv_1_t[0,15]
   = -1.04077e-08
 Conv_1_output_constraint[1,0]: 1.69823e-06 Conv_0_Relu_ts[1,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[1,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[1,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[1,12]
   + Conv_1_t[1,0] = -1.7536e-08
 Conv_1_output_constraint[1,1]: 6.64218e-05 Conv_0_Relu_ts[1,0]
   - 1.304573059082031 Conv_0_Relu_ts[1,1]
   - 0.00808643829077482 Conv_0_Relu_ts[1,2]
   + 1.10233e-12 Conv_0_Relu_ts[1,4] - 6.46578e-08 Conv_0_Relu_ts[1,6]
   + 3.59547e-13 Conv_0_Relu_ts[1,7] + 2.26512e-09 Conv_0_Relu_ts[1,8]
   - 1.48387e-10 Conv_0_Relu_ts[1,9]
   - 0.0375515259802341 Conv_0_Relu_ts[1,10]
   - 0.1779745519161224 Conv_0_Relu_ts[1,12]
   - 1.40572e-06 Conv_0_Relu_ts[1,13]
   + 4.89468e-05 Conv_1_aggregated_features[1,0]
   - 0.00799453444778919 Conv_1_aggregated_features[1,1]
   - 0.0869906693696976 Conv_1_aggregated_features[1,2]
   - 3.80301e-08 Conv_1_aggregated_features[1,6]
   + 2.46415e-09 Conv_1_aggregated_features[1,8]
   - 1.62219e-10 Conv_1_aggregated_features[1,9]
   - 0.2601335346698761 Conv_1_aggregated_features[1,10]
   + 0.4667477905750275 Conv_1_aggregated_features[1,12]
   - 1.71701e-06 Conv_1_aggregated_features[1,13] + Conv_1_t[1,1]
   = 0.3359648287296295
 Conv_1_output_constraint[1,2]: - 2.68276e-05 Conv_0_Relu_ts[1,0]
   + 7.4485e-11 Conv_0_Relu_ts[1,1] + 3.46068e-11 Conv_0_Relu_ts[1,2]
   - 5.94579e-10 Conv_0_Relu_ts[1,4] + 1.49975e-09 Conv_0_Relu_ts[1,5]
   + 4.00629e-09 Conv_0_Relu_ts[1,6] - 0.000385428 Conv_0_Relu_ts[1,8]
   - 9.65733e-05 Conv_0_Relu_ts[1,9] + 3.04566e-10 Conv_0_Relu_ts[1,10]
   + 2.67322e-05 Conv_0_Relu_ts[1,11] + 5.13886e-09 Conv_0_Relu_ts[1,12]
   - 6.79858e-13 Conv_0_Relu_ts[1,13]
   + 2.85913e-06 Conv_1_aggregated_features[1,0]
   + 3.02975e-09 Conv_1_aggregated_features[1,1]
   + 6.8068e-11 Conv_1_aggregated_features[1,2]
   + 2.87298e-07 Conv_1_aggregated_features[1,4]
   + 1.28094e-07 Conv_1_aggregated_features[1,5]
   + 3.03159e-11 Conv_1_aggregated_features[1,6]
   + 2.96378e-05 Conv_1_aggregated_features[1,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[1,9]
   + 7.60786e-11 Conv_1_aggregated_features[1,10]
   + 9.73847e-08 Conv_1_aggregated_features[1,11]
   + 7.54748e-09 Conv_1_aggregated_features[1,12]
   + 1.58189e-09 Conv_1_aggregated_features[1,13] + Conv_1_t[1,2]
   = -1.53413e-08
 Conv_1_output_constraint[1,3]: 9.55069e-08 Conv_0_Relu_ts[1,3]
   - 4.06432e-09 Conv_0_Relu_ts[1,5] - 2.23188e-09 Conv_0_Relu_ts[1,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[1,12]
   + 4.18428e-06 Conv_0_Relu_ts[1,13]
   + 7.14864e-07 Conv_1_aggregated_features[1,3]
   - 9.90425e-11 Conv_1_aggregated_features[1,5]
   + 2.85992e-07 Conv_1_aggregated_features[1,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[1,12]
   - 7.41084e-08 Conv_1_aggregated_features[1,13] + Conv_1_t[1,3]
   = -3.15407e-06
 Conv_1_output_constraint[1,4]: 2.97121e-10 Conv_0_Relu_ts[1,0]
   + 9.07965e-08 Conv_0_Relu_ts[1,1] + 1.13683e-08 Conv_0_Relu_ts[1,2]
   + 7.68037e-08 Conv_0_Relu_ts[1,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[1,5]
   - 5.39198e-07 Conv_0_Relu_ts[1,6] - 6.18604e-12 Conv_0_Relu_ts[1,8]
   + 3.00799e-08 Conv_0_Relu_ts[1,10] - 1.66347e-05 Conv_0_Relu_ts[1,11]
   + 2.56472e-07 Conv_0_Relu_ts[1,12] - 3.27559e-05 Conv_0_Relu_ts[1,13]
   - 6.3717e-09 Conv_1_aggregated_features[1,0]
   + 3.53659e-08 Conv_1_aggregated_features[1,1]
   + 7.41964e-09 Conv_1_aggregated_features[1,2]
   + 0.00146958860568702 Conv_1_aggregated_features[1,4]
   + 1.41886e-06 Conv_1_aggregated_features[1,5]
   - 1.75898e-05 Conv_1_aggregated_features[1,6]
   + 0.0017772619612515 Conv_1_aggregated_features[1,8]
   + 7.64123e-10 Conv_1_aggregated_features[1,9]
   + 2.4839e-08 Conv_1_aggregated_features[1,10]
   - 2.68593e-07 Conv_1_aggregated_features[1,11]
   + 2.04143e-07 Conv_1_aggregated_features[1,12]
   + 4.75765e-08 Conv_1_aggregated_features[1,13] + Conv_1_t[1,4]
   = -8.58295e-08
 Conv_1_output_constraint[1,5]: - 1.12261e-07 Conv_0_Relu_ts[1,0]
   - 6.16837e-11 Conv_0_Relu_ts[1,1] - 1.43211e-10 Conv_0_Relu_ts[1,2]
   + 9.76965e-10 Conv_0_Relu_ts[1,3] - 8.36344e-05 Conv_0_Relu_ts[1,4]
   + 2.12422e-08 Conv_0_Relu_ts[1,5] - 2.04737e-07 Conv_0_Relu_ts[1,6]
   - 6.84967e-06 Conv_0_Relu_ts[1,8] + 2.60175e-05 Conv_0_Relu_ts[1,9]
   + 1.5048e-10 Conv_0_Relu_ts[1,10] - 2.26739e-07 Conv_0_Relu_ts[1,11]
   + 2.04031e-09 Conv_0_Relu_ts[1,12] + 3.64928e-12 Conv_0_Relu_ts[1,13]
   - 4.59272e-08 Conv_0_Relu_ts[1,15]
   + 8.09056e-08 Conv_1_aggregated_features[1,0]
   + 1.22547e-10 Conv_1_aggregated_features[1,1]
   + 1.29284e-10 Conv_1_aggregated_features[1,2]
   + 5.76885e-11 Conv_1_aggregated_features[1,3]
   + 5.76747e-08 Conv_1_aggregated_features[1,4]
   + 1.49451e-09 Conv_1_aggregated_features[1,5]
   + 9.85149e-10 Conv_1_aggregated_features[1,8]
   + 1.88005e-11 Conv_1_aggregated_features[1,9]
   + 3.09409e-11 Conv_1_aggregated_features[1,10]
   + 1.45468e-07 Conv_1_aggregated_features[1,11]
   + 2.76313e-09 Conv_1_aggregated_features[1,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[1,13]
   + Conv_1_t[1,5] = -4.66972e-09
 Conv_1_output_constraint[1,6]: 1.68259e-08 Conv_0_Relu_ts[1,0]
   + 2.66449e-06 Conv_0_Relu_ts[1,1] - 3.07804e-06 Conv_0_Relu_ts[1,2]
   - 1.56854e-10 Conv_0_Relu_ts[1,3] + 1.61577e-06 Conv_0_Relu_ts[1,4]
   + 9.63312e-13 Conv_0_Relu_ts[1,5] + 8.94799e-11 Conv_0_Relu_ts[1,6]
   + 0.00156858959235251 Conv_0_Relu_ts[1,8]
   + 3.71009e-11 Conv_0_Relu_ts[1,9] - 1.18851e-06 Conv_0_Relu_ts[1,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[1,11]
   + 2.08067e-05 Conv_0_Relu_ts[1,12] - 1.93794e-06 Conv_0_Relu_ts[1,13]
   + 2.00933e-08 Conv_1_aggregated_features[1,0]
   + 1.84936e-05 Conv_1_aggregated_features[1,1]
   - 2.40274e-06 Conv_1_aggregated_features[1,2]
   + 4.16152e-12 Conv_1_aggregated_features[1,3]
   - 9.5418e-06 Conv_1_aggregated_features[1,4]
   + 2.61288e-12 Conv_1_aggregated_features[1,5]
   - 5.0478e-05 Conv_1_aggregated_features[1,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[1,8]
   - 4.57774e-07 Conv_1_aggregated_features[1,9]
   - 7.00605e-07 Conv_1_aggregated_features[1,10]
   + 2.2587e-10 Conv_1_aggregated_features[1,11]
   + 4.15987e-05 Conv_1_aggregated_features[1,12]
   - 1.80554e-09 Conv_1_aggregated_features[1,13] + Conv_1_t[1,6]
   = -9.69776e-05
 Conv_1_output_constraint[1,7]: - 1.58234e-12 Conv_0_Relu_ts[1,0]
   + 6.43182e-10 Conv_0_Relu_ts[1,1] - 1.24733e-10 Conv_0_Relu_ts[1,2]
   + 7.71679e-06 Conv_0_Relu_ts[1,4] + 2.78803e-06 Conv_0_Relu_ts[1,6]
   + 2.03373e-06 Conv_0_Relu_ts[1,8] - 2.93243e-11 Conv_0_Relu_ts[1,9]
   - 2.23517e-10 Conv_0_Relu_ts[1,10] + 2.01832e-09 Conv_0_Relu_ts[1,12]
   + 5.32355e-07 Conv_1_aggregated_features[1,0]
   + 8.4797e-09 Conv_1_aggregated_features[1,1]
   - 1.50954e-10 Conv_1_aggregated_features[1,2]
   + 1.6387e-09 Conv_1_aggregated_features[1,4]
   + 2.45145e-06 Conv_1_aggregated_features[1,6]
   + 1.09862e-10 Conv_1_aggregated_features[1,7]
   - 1.68113e-05 Conv_1_aggregated_features[1,8]
   + 1.4449e-10 Conv_1_aggregated_features[1,9]
   - 4.62632e-10 Conv_1_aggregated_features[1,10]
   + 1.80231e-09 Conv_1_aggregated_features[1,12] + Conv_1_t[1,7]
   = -6.52043e-09
 Conv_1_output_constraint[1,8]: 2.0398e-13 Conv_1_aggregated_features[1,12]
   + Conv_1_t[1,8] = 2.40312e-18
 Conv_1_output_constraint[1,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[1,0]
   + 0.4666179716587067 Conv_0_Relu_ts[1,1]
   - 0.212794229388237 Conv_0_Relu_ts[1,2]
   + 1.31709e-13 Conv_0_Relu_ts[1,3] + 6.41139e-10 Conv_0_Relu_ts[1,4]
   - 1.36516e-10 Conv_0_Relu_ts[1,5]
   + 0.00123115978203714 Conv_0_Relu_ts[1,6]
   - 3.8707e-07 Conv_0_Relu_ts[1,7] - 9.99069e-10 Conv_0_Relu_ts[1,8]
   - 3.90229e-11 Conv_0_Relu_ts[1,9]
   - 0.3983311057090759 Conv_0_Relu_ts[1,10]
   + 1.4696e-09 Conv_0_Relu_ts[1,11]
   + 0.135991245508194 Conv_0_Relu_ts[1,12]
   - 5.03958e-07 Conv_0_Relu_ts[1,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[1,0]
   + 0.0555339716374874 Conv_1_aggregated_features[1,1]
   - 0.1152108833193779 Conv_1_aggregated_features[1,2]
   + 1.01018e-09 Conv_1_aggregated_features[1,4]
   - 2.65221e-10 Conv_1_aggregated_features[1,5]
   + 0.00134999 Conv_1_aggregated_features[1,6]
   - 7.7429e-07 Conv_1_aggregated_features[1,7]
   - 9.0928e-10 Conv_1_aggregated_features[1,8]
   - 4.02666e-11 Conv_1_aggregated_features[1,9]
   - 0.1311033517122269 Conv_1_aggregated_features[1,10]
   + 5.72141e-09 Conv_1_aggregated_features[1,11]
   - 1.062317252159119 Conv_1_aggregated_features[1,12]
   - 3.93908e-07 Conv_1_aggregated_features[1,13] + Conv_1_t[1,9]
   = 0.4778489172458649
 Conv_1_output_constraint[1,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[1,0]
   + 3.12754e-07 Conv_0_Relu_ts[1,1] + 1.78384e-08 Conv_0_Relu_ts[1,2]
   - 8.1105e-12 Conv_0_Relu_ts[1,3] - 1.21368e-06 Conv_0_Relu_ts[1,4]
   - 7.58218e-10 Conv_0_Relu_ts[1,5] + 6.37659e-05 Conv_0_Relu_ts[1,7]
   + 5.06994e-09 Conv_0_Relu_ts[1,8] + 4.38429e-08 Conv_0_Relu_ts[1,10]
   + 1.00038e-06 Conv_0_Relu_ts[1,12] - 1.00897e-07 Conv_0_Relu_ts[1,13]
   - 7.72898e-10 Conv_1_aggregated_features[1,0]
   + 1.93344e-07 Conv_1_aggregated_features[1,1]
   + 2.31957e-08 Conv_1_aggregated_features[1,2]
   - 5.28535e-07 Conv_1_aggregated_features[1,3]
   - 8.50617e-06 Conv_1_aggregated_features[1,4]
   - 1.86843e-09 Conv_1_aggregated_features[1,5]
   - 4.42334e-07 Conv_1_aggregated_features[1,8]
   - 1.66725e-07 Conv_1_aggregated_features[1,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[1,11]
   + 9.07032e-07 Conv_1_aggregated_features[1,12]
   + 1.19635e-11 Conv_1_aggregated_features[1,13] + Conv_1_t[1,10]
   = -7.05563e-07
 Conv_1_output_constraint[1,11]: 1.34578e-06 Conv_0_Relu_ts[1,0]
   + 2.25868e-05 Conv_0_Relu_ts[1,1] + 1.36528e-05 Conv_0_Relu_ts[1,2]
   + 6.61139e-05 Conv_0_Relu_ts[1,4] + 1.97151e-05 Conv_0_Relu_ts[1,5]
   - 2.52237e-09 Conv_0_Relu_ts[1,8] - 2.08745e-05 Conv_0_Relu_ts[1,9]
   + 4.27105e-05 Conv_0_Relu_ts[1,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[1,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[1,12]
   + 9.27492e-09 Conv_0_Relu_ts[1,13] - 1.59817e-08 Conv_0_Relu_ts[1,15]
   - 7.80059e-05 Conv_1_aggregated_features[1,0]
   + 3.38276e-05 Conv_1_aggregated_features[1,1]
   + 4.73819e-06 Conv_1_aggregated_features[1,2]
   - 0.00209861318580806 Conv_1_aggregated_features[1,4]
   - 4.14211e-05 Conv_1_aggregated_features[1,5]
   - 1.65027e-11 Conv_1_aggregated_features[1,8]
   + 2.09826e-05 Conv_1_aggregated_features[1,10]
   + 3.24145e-10 Conv_1_aggregated_features[1,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[1,12]
   - 6.08701e-11 Conv_1_aggregated_features[1,13] + Conv_1_t[1,11]
   = 3.07161e-05
 Conv_1_output_constraint[1,12]: 7.60796e-09 Conv_0_Relu_ts[1,1]
   + 2.00773e-10 Conv_0_Relu_ts[1,2] + 3.86788e-10 Conv_0_Relu_ts[1,3]
   + 3.1184e-11 Conv_0_Relu_ts[1,8] + 1.83806e-10 Conv_0_Relu_ts[1,10]
   + 2.13717e-08 Conv_0_Relu_ts[1,12] - 6.30213e-12 Conv_0_Relu_ts[1,13]
   + 2.90536e-09 Conv_1_aggregated_features[1,1]
   - 3.16398e-10 Conv_1_aggregated_features[1,2]
   - 2.02197e-09 Conv_1_aggregated_features[1,3]
   - 1.19877e-11 Conv_1_aggregated_features[1,8]
   - 5.09457e-10 Conv_1_aggregated_features[1,10]
   + 2.27248e-08 Conv_1_aggregated_features[1,12]
   - 1.24645e-13 Conv_1_aggregated_features[1,13] + Conv_1_t[1,12]
   = -3.75308e-09
 Conv_1_output_constraint[1,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[1,0]
   - 0.1570860594511032 Conv_0_Relu_ts[1,1]
   + 0.3520863950252533 Conv_0_Relu_ts[1,2]
   - 1.29036e-08 Conv_0_Relu_ts[1,4] + 3.83205e-07 Conv_0_Relu_ts[1,5]
   - 0.00537013821303844 Conv_0_Relu_ts[1,6]
   - 8.97018e-08 Conv_0_Relu_ts[1,7] + 1.712e-10 Conv_0_Relu_ts[1,8]
   + 1.81878e-10 Conv_0_Relu_ts[1,9]
   + 0.4303593337535858 Conv_0_Relu_ts[1,10]
   - 1.61554e-08 Conv_0_Relu_ts[1,11]
   - 0.9454745650291443 Conv_0_Relu_ts[1,12]
   + 4.84941e-06 Conv_0_Relu_ts[1,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[1,0]
   - 0.7010135054588318 Conv_1_aggregated_features[1,1]
   + 0.0964145809412003 Conv_1_aggregated_features[1,2]
   - 3.74041e-08 Conv_1_aggregated_features[1,4]
   + 5.47003e-07 Conv_1_aggregated_features[1,5]
   - 0.00514170713722706 Conv_1_aggregated_features[1,6]
   - 9.53448e-08 Conv_1_aggregated_features[1,7]
   - 2.07781e-10 Conv_1_aggregated_features[1,8]
   + 1.66638e-10 Conv_1_aggregated_features[1,9]
   + 0.1311885118484497 Conv_1_aggregated_features[1,10]
   - 9.23591e-08 Conv_1_aggregated_features[1,11]
   - 0.1851733475923538 Conv_1_aggregated_features[1,12]
   + 5.0438e-06 Conv_1_aggregated_features[1,13] + Conv_1_t[1,13]
   = -1.276697397232056
 Conv_1_output_constraint[1,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[1,0]
   + 0.00206982018426061 Conv_0_Relu_ts[1,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[1,2]
   - 9.77907e-13 Conv_0_Relu_ts[1,3] - 3.82624e-07 Conv_0_Relu_ts[1,4]
   + 1.19511e-07 Conv_0_Relu_ts[1,5] - 2.54839e-11 Conv_0_Relu_ts[1,6]
   + 4.99102e-07 Conv_0_Relu_ts[1,8] + 2.64257e-11 Conv_0_Relu_ts[1,9]
   - 0.00118294754065573 Conv_0_Relu_ts[1,10]
   + 4.8915e-08 Conv_0_Relu_ts[1,11]
   + 0.00699353078380227 Conv_0_Relu_ts[1,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[1,13]
   + 1.51033e-06 Conv_1_aggregated_features[1,0]
   + 0.00578453531488776 Conv_1_aggregated_features[1,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[1,2]
   + 5.14278e-09 Conv_1_aggregated_features[1,4]
   + 3.41007e-10 Conv_1_aggregated_features[1,5]
   + 1.47387e-08 Conv_1_aggregated_features[1,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[1,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[1,10]
   + 1.3873e-07 Conv_1_aggregated_features[1,11]
   + 0.0143724931403995 Conv_1_aggregated_features[1,12]
   - 6.50481e-05 Conv_1_aggregated_features[1,13] + Conv_1_t[1,14]
   = -0.00573839386925101
 Conv_1_output_constraint[1,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[1,0]
   + 3.59972e-09 Conv_0_Relu_ts[1,1] - 2.63862e-12 Conv_0_Relu_ts[1,2]
   - 0.00124379759654403 Conv_0_Relu_ts[1,4]
   + 1.90933e-08 Conv_0_Relu_ts[1,5] - 4.63531e-13 Conv_0_Relu_ts[1,6]
   - 1.54557e-05 Conv_0_Relu_ts[1,7] - 1.60225e-10 Conv_0_Relu_ts[1,8]
   + 2.49673e-06 Conv_0_Relu_ts[1,9] + 7.17346e-12 Conv_0_Relu_ts[1,10]
   - 3.53936e-11 Conv_0_Relu_ts[1,11] + 1.06168e-09 Conv_0_Relu_ts[1,12]
   - 4.87679e-07 Conv_0_Relu_ts[1,13] + 3.14464e-11 Conv_0_Relu_ts[1,15]
   - 0.000458086 Conv_1_aggregated_features[1,0]
   + 7.50942e-09 Conv_1_aggregated_features[1,1]
   - 3.13868e-12 Conv_1_aggregated_features[1,2]
   - 5.10815e-06 Conv_1_aggregated_features[1,4]
   - 1.93728e-06 Conv_1_aggregated_features[1,5]
   - 9.15832e-12 Conv_1_aggregated_features[1,6]
   + 5.1922e-09 Conv_1_aggregated_features[1,7]
   + 2.54628e-11 Conv_1_aggregated_features[1,8]
   - 1.10428e-11 Conv_1_aggregated_features[1,10]
   + 1.00597e-09 Conv_1_aggregated_features[1,12]
   + 7.86155e-08 Conv_1_aggregated_features[1,13] + Conv_1_t[1,15]
   = -1.04077e-08
 Conv_1_output_constraint[2,0]: 1.69823e-06 Conv_0_Relu_ts[2,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[2,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[2,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[2,12]
   + Conv_1_t[2,0] = -1.7536e-08
 Conv_1_output_constraint[2,1]: 6.64218e-05 Conv_0_Relu_ts[2,0]
   - 1.304573059082031 Conv_0_Relu_ts[2,1]
   - 0.00808643829077482 Conv_0_Relu_ts[2,2]
   + 1.10233e-12 Conv_0_Relu_ts[2,4] - 6.46578e-08 Conv_0_Relu_ts[2,6]
   + 3.59547e-13 Conv_0_Relu_ts[2,7] + 2.26512e-09 Conv_0_Relu_ts[2,8]
   - 1.48387e-10 Conv_0_Relu_ts[2,9]
   - 0.0375515259802341 Conv_0_Relu_ts[2,10]
   - 0.1779745519161224 Conv_0_Relu_ts[2,12]
   - 1.40572e-06 Conv_0_Relu_ts[2,13]
   + 4.89468e-05 Conv_1_aggregated_features[2,0]
   - 0.00799453444778919 Conv_1_aggregated_features[2,1]
   - 0.0869906693696976 Conv_1_aggregated_features[2,2]
   - 3.80301e-08 Conv_1_aggregated_features[2,6]
   + 2.46415e-09 Conv_1_aggregated_features[2,8]
   - 1.62219e-10 Conv_1_aggregated_features[2,9]
   - 0.2601335346698761 Conv_1_aggregated_features[2,10]
   + 0.4667477905750275 Conv_1_aggregated_features[2,12]
   - 1.71701e-06 Conv_1_aggregated_features[2,13] + Conv_1_t[2,1]
   = 0.3359648287296295
 Conv_1_output_constraint[2,2]: - 2.68276e-05 Conv_0_Relu_ts[2,0]
   + 7.4485e-11 Conv_0_Relu_ts[2,1] + 3.46068e-11 Conv_0_Relu_ts[2,2]
   - 5.94579e-10 Conv_0_Relu_ts[2,4] + 1.49975e-09 Conv_0_Relu_ts[2,5]
   + 4.00629e-09 Conv_0_Relu_ts[2,6] - 0.000385428 Conv_0_Relu_ts[2,8]
   - 9.65733e-05 Conv_0_Relu_ts[2,9] + 3.04566e-10 Conv_0_Relu_ts[2,10]
   + 2.67322e-05 Conv_0_Relu_ts[2,11] + 5.13886e-09 Conv_0_Relu_ts[2,12]
   - 6.79858e-13 Conv_0_Relu_ts[2,13]
   + 2.85913e-06 Conv_1_aggregated_features[2,0]
   + 3.02975e-09 Conv_1_aggregated_features[2,1]
   + 6.8068e-11 Conv_1_aggregated_features[2,2]
   + 2.87298e-07 Conv_1_aggregated_features[2,4]
   + 1.28094e-07 Conv_1_aggregated_features[2,5]
   + 3.03159e-11 Conv_1_aggregated_features[2,6]
   + 2.96378e-05 Conv_1_aggregated_features[2,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[2,9]
   + 7.60786e-11 Conv_1_aggregated_features[2,10]
   + 9.73847e-08 Conv_1_aggregated_features[2,11]
   + 7.54748e-09 Conv_1_aggregated_features[2,12]
   + 1.58189e-09 Conv_1_aggregated_features[2,13] + Conv_1_t[2,2]
   = -1.53413e-08
 Conv_1_output_constraint[2,3]: 9.55069e-08 Conv_0_Relu_ts[2,3]
   - 4.06432e-09 Conv_0_Relu_ts[2,5] - 2.23188e-09 Conv_0_Relu_ts[2,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[2,12]
   + 4.18428e-06 Conv_0_Relu_ts[2,13]
   + 7.14864e-07 Conv_1_aggregated_features[2,3]
   - 9.90425e-11 Conv_1_aggregated_features[2,5]
   + 2.85992e-07 Conv_1_aggregated_features[2,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[2,12]
   - 7.41084e-08 Conv_1_aggregated_features[2,13] + Conv_1_t[2,3]
   = -3.15407e-06
 Conv_1_output_constraint[2,4]: 2.97121e-10 Conv_0_Relu_ts[2,0]
   + 9.07965e-08 Conv_0_Relu_ts[2,1] + 1.13683e-08 Conv_0_Relu_ts[2,2]
   + 7.68037e-08 Conv_0_Relu_ts[2,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[2,5]
   - 5.39198e-07 Conv_0_Relu_ts[2,6] - 6.18604e-12 Conv_0_Relu_ts[2,8]
   + 3.00799e-08 Conv_0_Relu_ts[2,10] - 1.66347e-05 Conv_0_Relu_ts[2,11]
   + 2.56472e-07 Conv_0_Relu_ts[2,12] - 3.27559e-05 Conv_0_Relu_ts[2,13]
   - 6.3717e-09 Conv_1_aggregated_features[2,0]
   + 3.53659e-08 Conv_1_aggregated_features[2,1]
   + 7.41964e-09 Conv_1_aggregated_features[2,2]
   + 0.00146958860568702 Conv_1_aggregated_features[2,4]
   + 1.41886e-06 Conv_1_aggregated_features[2,5]
   - 1.75898e-05 Conv_1_aggregated_features[2,6]
   + 0.0017772619612515 Conv_1_aggregated_features[2,8]
   + 7.64123e-10 Conv_1_aggregated_features[2,9]
   + 2.4839e-08 Conv_1_aggregated_features[2,10]
   - 2.68593e-07 Conv_1_aggregated_features[2,11]
   + 2.04143e-07 Conv_1_aggregated_features[2,12]
   + 4.75765e-08 Conv_1_aggregated_features[2,13] + Conv_1_t[2,4]
   = -8.58295e-08
 Conv_1_output_constraint[2,5]: - 1.12261e-07 Conv_0_Relu_ts[2,0]
   - 6.16837e-11 Conv_0_Relu_ts[2,1] - 1.43211e-10 Conv_0_Relu_ts[2,2]
   + 9.76965e-10 Conv_0_Relu_ts[2,3] - 8.36344e-05 Conv_0_Relu_ts[2,4]
   + 2.12422e-08 Conv_0_Relu_ts[2,5] - 2.04737e-07 Conv_0_Relu_ts[2,6]
   - 6.84967e-06 Conv_0_Relu_ts[2,8] + 2.60175e-05 Conv_0_Relu_ts[2,9]
   + 1.5048e-10 Conv_0_Relu_ts[2,10] - 2.26739e-07 Conv_0_Relu_ts[2,11]
   + 2.04031e-09 Conv_0_Relu_ts[2,12] + 3.64928e-12 Conv_0_Relu_ts[2,13]
   - 4.59272e-08 Conv_0_Relu_ts[2,15]
   + 8.09056e-08 Conv_1_aggregated_features[2,0]
   + 1.22547e-10 Conv_1_aggregated_features[2,1]
   + 1.29284e-10 Conv_1_aggregated_features[2,2]
   + 5.76885e-11 Conv_1_aggregated_features[2,3]
   + 5.76747e-08 Conv_1_aggregated_features[2,4]
   + 1.49451e-09 Conv_1_aggregated_features[2,5]
   + 9.85149e-10 Conv_1_aggregated_features[2,8]
   + 1.88005e-11 Conv_1_aggregated_features[2,9]
   + 3.09409e-11 Conv_1_aggregated_features[2,10]
   + 1.45468e-07 Conv_1_aggregated_features[2,11]
   + 2.76313e-09 Conv_1_aggregated_features[2,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[2,13]
   + Conv_1_t[2,5] = -4.66972e-09
 Conv_1_output_constraint[2,6]: 1.68259e-08 Conv_0_Relu_ts[2,0]
   + 2.66449e-06 Conv_0_Relu_ts[2,1] - 3.07804e-06 Conv_0_Relu_ts[2,2]
   - 1.56854e-10 Conv_0_Relu_ts[2,3] + 1.61577e-06 Conv_0_Relu_ts[2,4]
   + 9.63312e-13 Conv_0_Relu_ts[2,5] + 8.94799e-11 Conv_0_Relu_ts[2,6]
   + 0.00156858959235251 Conv_0_Relu_ts[2,8]
   + 3.71009e-11 Conv_0_Relu_ts[2,9] - 1.18851e-06 Conv_0_Relu_ts[2,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[2,11]
   + 2.08067e-05 Conv_0_Relu_ts[2,12] - 1.93794e-06 Conv_0_Relu_ts[2,13]
   + 2.00933e-08 Conv_1_aggregated_features[2,0]
   + 1.84936e-05 Conv_1_aggregated_features[2,1]
   - 2.40274e-06 Conv_1_aggregated_features[2,2]
   + 4.16152e-12 Conv_1_aggregated_features[2,3]
   - 9.5418e-06 Conv_1_aggregated_features[2,4]
   + 2.61288e-12 Conv_1_aggregated_features[2,5]
   - 5.0478e-05 Conv_1_aggregated_features[2,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[2,8]
   - 4.57774e-07 Conv_1_aggregated_features[2,9]
   - 7.00605e-07 Conv_1_aggregated_features[2,10]
   + 2.2587e-10 Conv_1_aggregated_features[2,11]
   + 4.15987e-05 Conv_1_aggregated_features[2,12]
   - 1.80554e-09 Conv_1_aggregated_features[2,13] + Conv_1_t[2,6]
   = -9.69776e-05
 Conv_1_output_constraint[2,7]: - 1.58234e-12 Conv_0_Relu_ts[2,0]
   + 6.43182e-10 Conv_0_Relu_ts[2,1] - 1.24733e-10 Conv_0_Relu_ts[2,2]
   + 7.71679e-06 Conv_0_Relu_ts[2,4] + 2.78803e-06 Conv_0_Relu_ts[2,6]
   + 2.03373e-06 Conv_0_Relu_ts[2,8] - 2.93243e-11 Conv_0_Relu_ts[2,9]
   - 2.23517e-10 Conv_0_Relu_ts[2,10] + 2.01832e-09 Conv_0_Relu_ts[2,12]
   + 5.32355e-07 Conv_1_aggregated_features[2,0]
   + 8.4797e-09 Conv_1_aggregated_features[2,1]
   - 1.50954e-10 Conv_1_aggregated_features[2,2]
   + 1.6387e-09 Conv_1_aggregated_features[2,4]
   + 2.45145e-06 Conv_1_aggregated_features[2,6]
   + 1.09862e-10 Conv_1_aggregated_features[2,7]
   - 1.68113e-05 Conv_1_aggregated_features[2,8]
   + 1.4449e-10 Conv_1_aggregated_features[2,9]
   - 4.62632e-10 Conv_1_aggregated_features[2,10]
   + 1.80231e-09 Conv_1_aggregated_features[2,12] + Conv_1_t[2,7]
   = -6.52043e-09
 Conv_1_output_constraint[2,8]: 2.0398e-13 Conv_1_aggregated_features[2,12]
   + Conv_1_t[2,8] = 2.40312e-18
 Conv_1_output_constraint[2,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[2,0]
   + 0.4666179716587067 Conv_0_Relu_ts[2,1]
   - 0.212794229388237 Conv_0_Relu_ts[2,2]
   + 1.31709e-13 Conv_0_Relu_ts[2,3] + 6.41139e-10 Conv_0_Relu_ts[2,4]
   - 1.36516e-10 Conv_0_Relu_ts[2,5]
   + 0.00123115978203714 Conv_0_Relu_ts[2,6]
   - 3.8707e-07 Conv_0_Relu_ts[2,7] - 9.99069e-10 Conv_0_Relu_ts[2,8]
   - 3.90229e-11 Conv_0_Relu_ts[2,9]
   - 0.3983311057090759 Conv_0_Relu_ts[2,10]
   + 1.4696e-09 Conv_0_Relu_ts[2,11]
   + 0.135991245508194 Conv_0_Relu_ts[2,12]
   - 5.03958e-07 Conv_0_Relu_ts[2,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[2,0]
   + 0.0555339716374874 Conv_1_aggregated_features[2,1]
   - 0.1152108833193779 Conv_1_aggregated_features[2,2]
   + 1.01018e-09 Conv_1_aggregated_features[2,4]
   - 2.65221e-10 Conv_1_aggregated_features[2,5]
   + 0.00134999 Conv_1_aggregated_features[2,6]
   - 7.7429e-07 Conv_1_aggregated_features[2,7]
   - 9.0928e-10 Conv_1_aggregated_features[2,8]
   - 4.02666e-11 Conv_1_aggregated_features[2,9]
   - 0.1311033517122269 Conv_1_aggregated_features[2,10]
   + 5.72141e-09 Conv_1_aggregated_features[2,11]
   - 1.062317252159119 Conv_1_aggregated_features[2,12]
   - 3.93908e-07 Conv_1_aggregated_features[2,13] + Conv_1_t[2,9]
   = 0.4778489172458649
 Conv_1_output_constraint[2,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[2,0]
   + 3.12754e-07 Conv_0_Relu_ts[2,1] + 1.78384e-08 Conv_0_Relu_ts[2,2]
   - 8.1105e-12 Conv_0_Relu_ts[2,3] - 1.21368e-06 Conv_0_Relu_ts[2,4]
   - 7.58218e-10 Conv_0_Relu_ts[2,5] + 6.37659e-05 Conv_0_Relu_ts[2,7]
   + 5.06994e-09 Conv_0_Relu_ts[2,8] + 4.38429e-08 Conv_0_Relu_ts[2,10]
   + 1.00038e-06 Conv_0_Relu_ts[2,12] - 1.00897e-07 Conv_0_Relu_ts[2,13]
   - 7.72898e-10 Conv_1_aggregated_features[2,0]
   + 1.93344e-07 Conv_1_aggregated_features[2,1]
   + 2.31957e-08 Conv_1_aggregated_features[2,2]
   - 5.28535e-07 Conv_1_aggregated_features[2,3]
   - 8.50617e-06 Conv_1_aggregated_features[2,4]
   - 1.86843e-09 Conv_1_aggregated_features[2,5]
   - 4.42334e-07 Conv_1_aggregated_features[2,8]
   - 1.66725e-07 Conv_1_aggregated_features[2,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[2,11]
   + 9.07032e-07 Conv_1_aggregated_features[2,12]
   + 1.19635e-11 Conv_1_aggregated_features[2,13] + Conv_1_t[2,10]
   = -7.05563e-07
 Conv_1_output_constraint[2,11]: 1.34578e-06 Conv_0_Relu_ts[2,0]
   + 2.25868e-05 Conv_0_Relu_ts[2,1] + 1.36528e-05 Conv_0_Relu_ts[2,2]
   + 6.61139e-05 Conv_0_Relu_ts[2,4] + 1.97151e-05 Conv_0_Relu_ts[2,5]
   - 2.52237e-09 Conv_0_Relu_ts[2,8] - 2.08745e-05 Conv_0_Relu_ts[2,9]
   + 4.27105e-05 Conv_0_Relu_ts[2,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[2,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[2,12]
   + 9.27492e-09 Conv_0_Relu_ts[2,13] - 1.59817e-08 Conv_0_Relu_ts[2,15]
   - 7.80059e-05 Conv_1_aggregated_features[2,0]
   + 3.38276e-05 Conv_1_aggregated_features[2,1]
   + 4.73819e-06 Conv_1_aggregated_features[2,2]
   - 0.00209861318580806 Conv_1_aggregated_features[2,4]
   - 4.14211e-05 Conv_1_aggregated_features[2,5]
   - 1.65027e-11 Conv_1_aggregated_features[2,8]
   + 2.09826e-05 Conv_1_aggregated_features[2,10]
   + 3.24145e-10 Conv_1_aggregated_features[2,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[2,12]
   - 6.08701e-11 Conv_1_aggregated_features[2,13] + Conv_1_t[2,11]
   = 3.07161e-05
 Conv_1_output_constraint[2,12]: 7.60796e-09 Conv_0_Relu_ts[2,1]
   + 2.00773e-10 Conv_0_Relu_ts[2,2] + 3.86788e-10 Conv_0_Relu_ts[2,3]
   + 3.1184e-11 Conv_0_Relu_ts[2,8] + 1.83806e-10 Conv_0_Relu_ts[2,10]
   + 2.13717e-08 Conv_0_Relu_ts[2,12] - 6.30213e-12 Conv_0_Relu_ts[2,13]
   + 2.90536e-09 Conv_1_aggregated_features[2,1]
   - 3.16398e-10 Conv_1_aggregated_features[2,2]
   - 2.02197e-09 Conv_1_aggregated_features[2,3]
   - 1.19877e-11 Conv_1_aggregated_features[2,8]
   - 5.09457e-10 Conv_1_aggregated_features[2,10]
   + 2.27248e-08 Conv_1_aggregated_features[2,12]
   - 1.24645e-13 Conv_1_aggregated_features[2,13] + Conv_1_t[2,12]
   = -3.75308e-09
 Conv_1_output_constraint[2,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[2,0]
   - 0.1570860594511032 Conv_0_Relu_ts[2,1]
   + 0.3520863950252533 Conv_0_Relu_ts[2,2]
   - 1.29036e-08 Conv_0_Relu_ts[2,4] + 3.83205e-07 Conv_0_Relu_ts[2,5]
   - 0.00537013821303844 Conv_0_Relu_ts[2,6]
   - 8.97018e-08 Conv_0_Relu_ts[2,7] + 1.712e-10 Conv_0_Relu_ts[2,8]
   + 1.81878e-10 Conv_0_Relu_ts[2,9]
   + 0.4303593337535858 Conv_0_Relu_ts[2,10]
   - 1.61554e-08 Conv_0_Relu_ts[2,11]
   - 0.9454745650291443 Conv_0_Relu_ts[2,12]
   + 4.84941e-06 Conv_0_Relu_ts[2,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[2,0]
   - 0.7010135054588318 Conv_1_aggregated_features[2,1]
   + 0.0964145809412003 Conv_1_aggregated_features[2,2]
   - 3.74041e-08 Conv_1_aggregated_features[2,4]
   + 5.47003e-07 Conv_1_aggregated_features[2,5]
   - 0.00514170713722706 Conv_1_aggregated_features[2,6]
   - 9.53448e-08 Conv_1_aggregated_features[2,7]
   - 2.07781e-10 Conv_1_aggregated_features[2,8]
   + 1.66638e-10 Conv_1_aggregated_features[2,9]
   + 0.1311885118484497 Conv_1_aggregated_features[2,10]
   - 9.23591e-08 Conv_1_aggregated_features[2,11]
   - 0.1851733475923538 Conv_1_aggregated_features[2,12]
   + 5.0438e-06 Conv_1_aggregated_features[2,13] + Conv_1_t[2,13]
   = -1.276697397232056
 Conv_1_output_constraint[2,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[2,0]
   + 0.00206982018426061 Conv_0_Relu_ts[2,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[2,2]
   - 9.77907e-13 Conv_0_Relu_ts[2,3] - 3.82624e-07 Conv_0_Relu_ts[2,4]
   + 1.19511e-07 Conv_0_Relu_ts[2,5] - 2.54839e-11 Conv_0_Relu_ts[2,6]
   + 4.99102e-07 Conv_0_Relu_ts[2,8] + 2.64257e-11 Conv_0_Relu_ts[2,9]
   - 0.00118294754065573 Conv_0_Relu_ts[2,10]
   + 4.8915e-08 Conv_0_Relu_ts[2,11]
   + 0.00699353078380227 Conv_0_Relu_ts[2,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[2,13]
   + 1.51033e-06 Conv_1_aggregated_features[2,0]
   + 0.00578453531488776 Conv_1_aggregated_features[2,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[2,2]
   + 5.14278e-09 Conv_1_aggregated_features[2,4]
   + 3.41007e-10 Conv_1_aggregated_features[2,5]
   + 1.47387e-08 Conv_1_aggregated_features[2,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[2,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[2,10]
   + 1.3873e-07 Conv_1_aggregated_features[2,11]
   + 0.0143724931403995 Conv_1_aggregated_features[2,12]
   - 6.50481e-05 Conv_1_aggregated_features[2,13] + Conv_1_t[2,14]
   = -0.00573839386925101
 Conv_1_output_constraint[2,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[2,0]
   + 3.59972e-09 Conv_0_Relu_ts[2,1] - 2.63862e-12 Conv_0_Relu_ts[2,2]
   - 0.00124379759654403 Conv_0_Relu_ts[2,4]
   + 1.90933e-08 Conv_0_Relu_ts[2,5] - 4.63531e-13 Conv_0_Relu_ts[2,6]
   - 1.54557e-05 Conv_0_Relu_ts[2,7] - 1.60225e-10 Conv_0_Relu_ts[2,8]
   + 2.49673e-06 Conv_0_Relu_ts[2,9] + 7.17346e-12 Conv_0_Relu_ts[2,10]
   - 3.53936e-11 Conv_0_Relu_ts[2,11] + 1.06168e-09 Conv_0_Relu_ts[2,12]
   - 4.87679e-07 Conv_0_Relu_ts[2,13] + 3.14464e-11 Conv_0_Relu_ts[2,15]
   - 0.000458086 Conv_1_aggregated_features[2,0]
   + 7.50942e-09 Conv_1_aggregated_features[2,1]
   - 3.13868e-12 Conv_1_aggregated_features[2,2]
   - 5.10815e-06 Conv_1_aggregated_features[2,4]
   - 1.93728e-06 Conv_1_aggregated_features[2,5]
   - 9.15832e-12 Conv_1_aggregated_features[2,6]
   + 5.1922e-09 Conv_1_aggregated_features[2,7]
   + 2.54628e-11 Conv_1_aggregated_features[2,8]
   - 1.10428e-11 Conv_1_aggregated_features[2,10]
   + 1.00597e-09 Conv_1_aggregated_features[2,12]
   + 7.86155e-08 Conv_1_aggregated_features[2,13] + Conv_1_t[2,15]
   = -1.04077e-08
 Conv_1_output_constraint[3,0]: 1.69823e-06 Conv_0_Relu_ts[3,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[3,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[3,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[3,12]
   + Conv_1_t[3,0] = -1.7536e-08
 Conv_1_output_constraint[3,1]: 6.64218e-05 Conv_0_Relu_ts[3,0]
   - 1.304573059082031 Conv_0_Relu_ts[3,1]
   - 0.00808643829077482 Conv_0_Relu_ts[3,2]
   + 1.10233e-12 Conv_0_Relu_ts[3,4] - 6.46578e-08 Conv_0_Relu_ts[3,6]
   + 3.59547e-13 Conv_0_Relu_ts[3,7] + 2.26512e-09 Conv_0_Relu_ts[3,8]
   - 1.48387e-10 Conv_0_Relu_ts[3,9]
   - 0.0375515259802341 Conv_0_Relu_ts[3,10]
   - 0.1779745519161224 Conv_0_Relu_ts[3,12]
   - 1.40572e-06 Conv_0_Relu_ts[3,13]
   + 4.89468e-05 Conv_1_aggregated_features[3,0]
   - 0.00799453444778919 Conv_1_aggregated_features[3,1]
   - 0.0869906693696976 Conv_1_aggregated_features[3,2]
   - 3.80301e-08 Conv_1_aggregated_features[3,6]
   + 2.46415e-09 Conv_1_aggregated_features[3,8]
   - 1.62219e-10 Conv_1_aggregated_features[3,9]
   - 0.2601335346698761 Conv_1_aggregated_features[3,10]
   + 0.4667477905750275 Conv_1_aggregated_features[3,12]
   - 1.71701e-06 Conv_1_aggregated_features[3,13] + Conv_1_t[3,1]
   = 0.3359648287296295
 Conv_1_output_constraint[3,2]: - 2.68276e-05 Conv_0_Relu_ts[3,0]
   + 7.4485e-11 Conv_0_Relu_ts[3,1] + 3.46068e-11 Conv_0_Relu_ts[3,2]
   - 5.94579e-10 Conv_0_Relu_ts[3,4] + 1.49975e-09 Conv_0_Relu_ts[3,5]
   + 4.00629e-09 Conv_0_Relu_ts[3,6] - 0.000385428 Conv_0_Relu_ts[3,8]
   - 9.65733e-05 Conv_0_Relu_ts[3,9] + 3.04566e-10 Conv_0_Relu_ts[3,10]
   + 2.67322e-05 Conv_0_Relu_ts[3,11] + 5.13886e-09 Conv_0_Relu_ts[3,12]
   - 6.79858e-13 Conv_0_Relu_ts[3,13]
   + 2.85913e-06 Conv_1_aggregated_features[3,0]
   + 3.02975e-09 Conv_1_aggregated_features[3,1]
   + 6.8068e-11 Conv_1_aggregated_features[3,2]
   + 2.87298e-07 Conv_1_aggregated_features[3,4]
   + 1.28094e-07 Conv_1_aggregated_features[3,5]
   + 3.03159e-11 Conv_1_aggregated_features[3,6]
   + 2.96378e-05 Conv_1_aggregated_features[3,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[3,9]
   + 7.60786e-11 Conv_1_aggregated_features[3,10]
   + 9.73847e-08 Conv_1_aggregated_features[3,11]
   + 7.54748e-09 Conv_1_aggregated_features[3,12]
   + 1.58189e-09 Conv_1_aggregated_features[3,13] + Conv_1_t[3,2]
   = -1.53413e-08
 Conv_1_output_constraint[3,3]: 9.55069e-08 Conv_0_Relu_ts[3,3]
   - 4.06432e-09 Conv_0_Relu_ts[3,5] - 2.23188e-09 Conv_0_Relu_ts[3,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[3,12]
   + 4.18428e-06 Conv_0_Relu_ts[3,13]
   + 7.14864e-07 Conv_1_aggregated_features[3,3]
   - 9.90425e-11 Conv_1_aggregated_features[3,5]
   + 2.85992e-07 Conv_1_aggregated_features[3,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[3,12]
   - 7.41084e-08 Conv_1_aggregated_features[3,13] + Conv_1_t[3,3]
   = -3.15407e-06
 Conv_1_output_constraint[3,4]: 2.97121e-10 Conv_0_Relu_ts[3,0]
   + 9.07965e-08 Conv_0_Relu_ts[3,1] + 1.13683e-08 Conv_0_Relu_ts[3,2]
   + 7.68037e-08 Conv_0_Relu_ts[3,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[3,5]
   - 5.39198e-07 Conv_0_Relu_ts[3,6] - 6.18604e-12 Conv_0_Relu_ts[3,8]
   + 3.00799e-08 Conv_0_Relu_ts[3,10] - 1.66347e-05 Conv_0_Relu_ts[3,11]
   + 2.56472e-07 Conv_0_Relu_ts[3,12] - 3.27559e-05 Conv_0_Relu_ts[3,13]
   - 6.3717e-09 Conv_1_aggregated_features[3,0]
   + 3.53659e-08 Conv_1_aggregated_features[3,1]
   + 7.41964e-09 Conv_1_aggregated_features[3,2]
   + 0.00146958860568702 Conv_1_aggregated_features[3,4]
   + 1.41886e-06 Conv_1_aggregated_features[3,5]
   - 1.75898e-05 Conv_1_aggregated_features[3,6]
   + 0.0017772619612515 Conv_1_aggregated_features[3,8]
   + 7.64123e-10 Conv_1_aggregated_features[3,9]
   + 2.4839e-08 Conv_1_aggregated_features[3,10]
   - 2.68593e-07 Conv_1_aggregated_features[3,11]
   + 2.04143e-07 Conv_1_aggregated_features[3,12]
   + 4.75765e-08 Conv_1_aggregated_features[3,13] + Conv_1_t[3,4]
   = -8.58295e-08
 Conv_1_output_constraint[3,5]: - 1.12261e-07 Conv_0_Relu_ts[3,0]
   - 6.16837e-11 Conv_0_Relu_ts[3,1] - 1.43211e-10 Conv_0_Relu_ts[3,2]
   + 9.76965e-10 Conv_0_Relu_ts[3,3] - 8.36344e-05 Conv_0_Relu_ts[3,4]
   + 2.12422e-08 Conv_0_Relu_ts[3,5] - 2.04737e-07 Conv_0_Relu_ts[3,6]
   - 6.84967e-06 Conv_0_Relu_ts[3,8] + 2.60175e-05 Conv_0_Relu_ts[3,9]
   + 1.5048e-10 Conv_0_Relu_ts[3,10] - 2.26739e-07 Conv_0_Relu_ts[3,11]
   + 2.04031e-09 Conv_0_Relu_ts[3,12] + 3.64928e-12 Conv_0_Relu_ts[3,13]
   - 4.59272e-08 Conv_0_Relu_ts[3,15]
   + 8.09056e-08 Conv_1_aggregated_features[3,0]
   + 1.22547e-10 Conv_1_aggregated_features[3,1]
   + 1.29284e-10 Conv_1_aggregated_features[3,2]
   + 5.76885e-11 Conv_1_aggregated_features[3,3]
   + 5.76747e-08 Conv_1_aggregated_features[3,4]
   + 1.49451e-09 Conv_1_aggregated_features[3,5]
   + 9.85149e-10 Conv_1_aggregated_features[3,8]
   + 1.88005e-11 Conv_1_aggregated_features[3,9]
   + 3.09409e-11 Conv_1_aggregated_features[3,10]
   + 1.45468e-07 Conv_1_aggregated_features[3,11]
   + 2.76313e-09 Conv_1_aggregated_features[3,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[3,13]
   + Conv_1_t[3,5] = -4.66972e-09
 Conv_1_output_constraint[3,6]: 1.68259e-08 Conv_0_Relu_ts[3,0]
   + 2.66449e-06 Conv_0_Relu_ts[3,1] - 3.07804e-06 Conv_0_Relu_ts[3,2]
   - 1.56854e-10 Conv_0_Relu_ts[3,3] + 1.61577e-06 Conv_0_Relu_ts[3,4]
   + 9.63312e-13 Conv_0_Relu_ts[3,5] + 8.94799e-11 Conv_0_Relu_ts[3,6]
   + 0.00156858959235251 Conv_0_Relu_ts[3,8]
   + 3.71009e-11 Conv_0_Relu_ts[3,9] - 1.18851e-06 Conv_0_Relu_ts[3,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[3,11]
   + 2.08067e-05 Conv_0_Relu_ts[3,12] - 1.93794e-06 Conv_0_Relu_ts[3,13]
   + 2.00933e-08 Conv_1_aggregated_features[3,0]
   + 1.84936e-05 Conv_1_aggregated_features[3,1]
   - 2.40274e-06 Conv_1_aggregated_features[3,2]
   + 4.16152e-12 Conv_1_aggregated_features[3,3]
   - 9.5418e-06 Conv_1_aggregated_features[3,4]
   + 2.61288e-12 Conv_1_aggregated_features[3,5]
   - 5.0478e-05 Conv_1_aggregated_features[3,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[3,8]
   - 4.57774e-07 Conv_1_aggregated_features[3,9]
   - 7.00605e-07 Conv_1_aggregated_features[3,10]
   + 2.2587e-10 Conv_1_aggregated_features[3,11]
   + 4.15987e-05 Conv_1_aggregated_features[3,12]
   - 1.80554e-09 Conv_1_aggregated_features[3,13] + Conv_1_t[3,6]
   = -9.69776e-05
 Conv_1_output_constraint[3,7]: - 1.58234e-12 Conv_0_Relu_ts[3,0]
   + 6.43182e-10 Conv_0_Relu_ts[3,1] - 1.24733e-10 Conv_0_Relu_ts[3,2]
   + 7.71679e-06 Conv_0_Relu_ts[3,4] + 2.78803e-06 Conv_0_Relu_ts[3,6]
   + 2.03373e-06 Conv_0_Relu_ts[3,8] - 2.93243e-11 Conv_0_Relu_ts[3,9]
   - 2.23517e-10 Conv_0_Relu_ts[3,10] + 2.01832e-09 Conv_0_Relu_ts[3,12]
   + 5.32355e-07 Conv_1_aggregated_features[3,0]
   + 8.4797e-09 Conv_1_aggregated_features[3,1]
   - 1.50954e-10 Conv_1_aggregated_features[3,2]
   + 1.6387e-09 Conv_1_aggregated_features[3,4]
   + 2.45145e-06 Conv_1_aggregated_features[3,6]
   + 1.09862e-10 Conv_1_aggregated_features[3,7]
   - 1.68113e-05 Conv_1_aggregated_features[3,8]
   + 1.4449e-10 Conv_1_aggregated_features[3,9]
   - 4.62632e-10 Conv_1_aggregated_features[3,10]
   + 1.80231e-09 Conv_1_aggregated_features[3,12] + Conv_1_t[3,7]
   = -6.52043e-09
 Conv_1_output_constraint[3,8]: 2.0398e-13 Conv_1_aggregated_features[3,12]
   + Conv_1_t[3,8] = 2.40312e-18
 Conv_1_output_constraint[3,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[3,0]
   + 0.4666179716587067 Conv_0_Relu_ts[3,1]
   - 0.212794229388237 Conv_0_Relu_ts[3,2]
   + 1.31709e-13 Conv_0_Relu_ts[3,3] + 6.41139e-10 Conv_0_Relu_ts[3,4]
   - 1.36516e-10 Conv_0_Relu_ts[3,5]
   + 0.00123115978203714 Conv_0_Relu_ts[3,6]
   - 3.8707e-07 Conv_0_Relu_ts[3,7] - 9.99069e-10 Conv_0_Relu_ts[3,8]
   - 3.90229e-11 Conv_0_Relu_ts[3,9]
   - 0.3983311057090759 Conv_0_Relu_ts[3,10]
   + 1.4696e-09 Conv_0_Relu_ts[3,11]
   + 0.135991245508194 Conv_0_Relu_ts[3,12]
   - 5.03958e-07 Conv_0_Relu_ts[3,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[3,0]
   + 0.0555339716374874 Conv_1_aggregated_features[3,1]
   - 0.1152108833193779 Conv_1_aggregated_features[3,2]
   + 1.01018e-09 Conv_1_aggregated_features[3,4]
   - 2.65221e-10 Conv_1_aggregated_features[3,5]
   + 0.00134999 Conv_1_aggregated_features[3,6]
   - 7.7429e-07 Conv_1_aggregated_features[3,7]
   - 9.0928e-10 Conv_1_aggregated_features[3,8]
   - 4.02666e-11 Conv_1_aggregated_features[3,9]
   - 0.1311033517122269 Conv_1_aggregated_features[3,10]
   + 5.72141e-09 Conv_1_aggregated_features[3,11]
   - 1.062317252159119 Conv_1_aggregated_features[3,12]
   - 3.93908e-07 Conv_1_aggregated_features[3,13] + Conv_1_t[3,9]
   = 0.4778489172458649
 Conv_1_output_constraint[3,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[3,0]
   + 3.12754e-07 Conv_0_Relu_ts[3,1] + 1.78384e-08 Conv_0_Relu_ts[3,2]
   - 8.1105e-12 Conv_0_Relu_ts[3,3] - 1.21368e-06 Conv_0_Relu_ts[3,4]
   - 7.58218e-10 Conv_0_Relu_ts[3,5] + 6.37659e-05 Conv_0_Relu_ts[3,7]
   + 5.06994e-09 Conv_0_Relu_ts[3,8] + 4.38429e-08 Conv_0_Relu_ts[3,10]
   + 1.00038e-06 Conv_0_Relu_ts[3,12] - 1.00897e-07 Conv_0_Relu_ts[3,13]
   - 7.72898e-10 Conv_1_aggregated_features[3,0]
   + 1.93344e-07 Conv_1_aggregated_features[3,1]
   + 2.31957e-08 Conv_1_aggregated_features[3,2]
   - 5.28535e-07 Conv_1_aggregated_features[3,3]
   - 8.50617e-06 Conv_1_aggregated_features[3,4]
   - 1.86843e-09 Conv_1_aggregated_features[3,5]
   - 4.42334e-07 Conv_1_aggregated_features[3,8]
   - 1.66725e-07 Conv_1_aggregated_features[3,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[3,11]
   + 9.07032e-07 Conv_1_aggregated_features[3,12]
   + 1.19635e-11 Conv_1_aggregated_features[3,13] + Conv_1_t[3,10]
   = -7.05563e-07
 Conv_1_output_constraint[3,11]: 1.34578e-06 Conv_0_Relu_ts[3,0]
   + 2.25868e-05 Conv_0_Relu_ts[3,1] + 1.36528e-05 Conv_0_Relu_ts[3,2]
   + 6.61139e-05 Conv_0_Relu_ts[3,4] + 1.97151e-05 Conv_0_Relu_ts[3,5]
   - 2.52237e-09 Conv_0_Relu_ts[3,8] - 2.08745e-05 Conv_0_Relu_ts[3,9]
   + 4.27105e-05 Conv_0_Relu_ts[3,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[3,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[3,12]
   + 9.27492e-09 Conv_0_Relu_ts[3,13] - 1.59817e-08 Conv_0_Relu_ts[3,15]
   - 7.80059e-05 Conv_1_aggregated_features[3,0]
   + 3.38276e-05 Conv_1_aggregated_features[3,1]
   + 4.73819e-06 Conv_1_aggregated_features[3,2]
   - 0.00209861318580806 Conv_1_aggregated_features[3,4]
   - 4.14211e-05 Conv_1_aggregated_features[3,5]
   - 1.65027e-11 Conv_1_aggregated_features[3,8]
   + 2.09826e-05 Conv_1_aggregated_features[3,10]
   + 3.24145e-10 Conv_1_aggregated_features[3,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[3,12]
   - 6.08701e-11 Conv_1_aggregated_features[3,13] + Conv_1_t[3,11]
   = 3.07161e-05
 Conv_1_output_constraint[3,12]: 7.60796e-09 Conv_0_Relu_ts[3,1]
   + 2.00773e-10 Conv_0_Relu_ts[3,2] + 3.86788e-10 Conv_0_Relu_ts[3,3]
   + 3.1184e-11 Conv_0_Relu_ts[3,8] + 1.83806e-10 Conv_0_Relu_ts[3,10]
   + 2.13717e-08 Conv_0_Relu_ts[3,12] - 6.30213e-12 Conv_0_Relu_ts[3,13]
   + 2.90536e-09 Conv_1_aggregated_features[3,1]
   - 3.16398e-10 Conv_1_aggregated_features[3,2]
   - 2.02197e-09 Conv_1_aggregated_features[3,3]
   - 1.19877e-11 Conv_1_aggregated_features[3,8]
   - 5.09457e-10 Conv_1_aggregated_features[3,10]
   + 2.27248e-08 Conv_1_aggregated_features[3,12]
   - 1.24645e-13 Conv_1_aggregated_features[3,13] + Conv_1_t[3,12]
   = -3.75308e-09
 Conv_1_output_constraint[3,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[3,0]
   - 0.1570860594511032 Conv_0_Relu_ts[3,1]
   + 0.3520863950252533 Conv_0_Relu_ts[3,2]
   - 1.29036e-08 Conv_0_Relu_ts[3,4] + 3.83205e-07 Conv_0_Relu_ts[3,5]
   - 0.00537013821303844 Conv_0_Relu_ts[3,6]
   - 8.97018e-08 Conv_0_Relu_ts[3,7] + 1.712e-10 Conv_0_Relu_ts[3,8]
   + 1.81878e-10 Conv_0_Relu_ts[3,9]
   + 0.4303593337535858 Conv_0_Relu_ts[3,10]
   - 1.61554e-08 Conv_0_Relu_ts[3,11]
   - 0.9454745650291443 Conv_0_Relu_ts[3,12]
   + 4.84941e-06 Conv_0_Relu_ts[3,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[3,0]
   - 0.7010135054588318 Conv_1_aggregated_features[3,1]
   + 0.0964145809412003 Conv_1_aggregated_features[3,2]
   - 3.74041e-08 Conv_1_aggregated_features[3,4]
   + 5.47003e-07 Conv_1_aggregated_features[3,5]
   - 0.00514170713722706 Conv_1_aggregated_features[3,6]
   - 9.53448e-08 Conv_1_aggregated_features[3,7]
   - 2.07781e-10 Conv_1_aggregated_features[3,8]
   + 1.66638e-10 Conv_1_aggregated_features[3,9]
   + 0.1311885118484497 Conv_1_aggregated_features[3,10]
   - 9.23591e-08 Conv_1_aggregated_features[3,11]
   - 0.1851733475923538 Conv_1_aggregated_features[3,12]
   + 5.0438e-06 Conv_1_aggregated_features[3,13] + Conv_1_t[3,13]
   = -1.276697397232056
 Conv_1_output_constraint[3,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[3,0]
   + 0.00206982018426061 Conv_0_Relu_ts[3,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[3,2]
   - 9.77907e-13 Conv_0_Relu_ts[3,3] - 3.82624e-07 Conv_0_Relu_ts[3,4]
   + 1.19511e-07 Conv_0_Relu_ts[3,5] - 2.54839e-11 Conv_0_Relu_ts[3,6]
   + 4.99102e-07 Conv_0_Relu_ts[3,8] + 2.64257e-11 Conv_0_Relu_ts[3,9]
   - 0.00118294754065573 Conv_0_Relu_ts[3,10]
   + 4.8915e-08 Conv_0_Relu_ts[3,11]
   + 0.00699353078380227 Conv_0_Relu_ts[3,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[3,13]
   + 1.51033e-06 Conv_1_aggregated_features[3,0]
   + 0.00578453531488776 Conv_1_aggregated_features[3,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[3,2]
   + 5.14278e-09 Conv_1_aggregated_features[3,4]
   + 3.41007e-10 Conv_1_aggregated_features[3,5]
   + 1.47387e-08 Conv_1_aggregated_features[3,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[3,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[3,10]
   + 1.3873e-07 Conv_1_aggregated_features[3,11]
   + 0.0143724931403995 Conv_1_aggregated_features[3,12]
   - 6.50481e-05 Conv_1_aggregated_features[3,13] + Conv_1_t[3,14]
   = -0.00573839386925101
 Conv_1_output_constraint[3,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[3,0]
   + 3.59972e-09 Conv_0_Relu_ts[3,1] - 2.63862e-12 Conv_0_Relu_ts[3,2]
   - 0.00124379759654403 Conv_0_Relu_ts[3,4]
   + 1.90933e-08 Conv_0_Relu_ts[3,5] - 4.63531e-13 Conv_0_Relu_ts[3,6]
   - 1.54557e-05 Conv_0_Relu_ts[3,7] - 1.60225e-10 Conv_0_Relu_ts[3,8]
   + 2.49673e-06 Conv_0_Relu_ts[3,9] + 7.17346e-12 Conv_0_Relu_ts[3,10]
   - 3.53936e-11 Conv_0_Relu_ts[3,11] + 1.06168e-09 Conv_0_Relu_ts[3,12]
   - 4.87679e-07 Conv_0_Relu_ts[3,13] + 3.14464e-11 Conv_0_Relu_ts[3,15]
   - 0.000458086 Conv_1_aggregated_features[3,0]
   + 7.50942e-09 Conv_1_aggregated_features[3,1]
   - 3.13868e-12 Conv_1_aggregated_features[3,2]
   - 5.10815e-06 Conv_1_aggregated_features[3,4]
   - 1.93728e-06 Conv_1_aggregated_features[3,5]
   - 9.15832e-12 Conv_1_aggregated_features[3,6]
   + 5.1922e-09 Conv_1_aggregated_features[3,7]
   + 2.54628e-11 Conv_1_aggregated_features[3,8]
   - 1.10428e-11 Conv_1_aggregated_features[3,10]
   + 1.00597e-09 Conv_1_aggregated_features[3,12]
   + 7.86155e-08 Conv_1_aggregated_features[3,13] + Conv_1_t[3,15]
   = -1.04077e-08
 Conv_1_output_constraint[4,0]: 1.69823e-06 Conv_0_Relu_ts[4,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[4,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[4,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[4,12]
   + Conv_1_t[4,0] = -1.7536e-08
 Conv_1_output_constraint[4,1]: 6.64218e-05 Conv_0_Relu_ts[4,0]
   - 1.304573059082031 Conv_0_Relu_ts[4,1]
   - 0.00808643829077482 Conv_0_Relu_ts[4,2]
   + 1.10233e-12 Conv_0_Relu_ts[4,4] - 6.46578e-08 Conv_0_Relu_ts[4,6]
   + 3.59547e-13 Conv_0_Relu_ts[4,7] + 2.26512e-09 Conv_0_Relu_ts[4,8]
   - 1.48387e-10 Conv_0_Relu_ts[4,9]
   - 0.0375515259802341 Conv_0_Relu_ts[4,10]
   - 0.1779745519161224 Conv_0_Relu_ts[4,12]
   - 1.40572e-06 Conv_0_Relu_ts[4,13]
   + 4.89468e-05 Conv_1_aggregated_features[4,0]
   - 0.00799453444778919 Conv_1_aggregated_features[4,1]
   - 0.0869906693696976 Conv_1_aggregated_features[4,2]
   - 3.80301e-08 Conv_1_aggregated_features[4,6]
   + 2.46415e-09 Conv_1_aggregated_features[4,8]
   - 1.62219e-10 Conv_1_aggregated_features[4,9]
   - 0.2601335346698761 Conv_1_aggregated_features[4,10]
   + 0.4667477905750275 Conv_1_aggregated_features[4,12]
   - 1.71701e-06 Conv_1_aggregated_features[4,13] + Conv_1_t[4,1]
   = 0.3359648287296295
 Conv_1_output_constraint[4,2]: - 2.68276e-05 Conv_0_Relu_ts[4,0]
   + 7.4485e-11 Conv_0_Relu_ts[4,1] + 3.46068e-11 Conv_0_Relu_ts[4,2]
   - 5.94579e-10 Conv_0_Relu_ts[4,4] + 1.49975e-09 Conv_0_Relu_ts[4,5]
   + 4.00629e-09 Conv_0_Relu_ts[4,6] - 0.000385428 Conv_0_Relu_ts[4,8]
   - 9.65733e-05 Conv_0_Relu_ts[4,9] + 3.04566e-10 Conv_0_Relu_ts[4,10]
   + 2.67322e-05 Conv_0_Relu_ts[4,11] + 5.13886e-09 Conv_0_Relu_ts[4,12]
   - 6.79858e-13 Conv_0_Relu_ts[4,13]
   + 2.85913e-06 Conv_1_aggregated_features[4,0]
   + 3.02975e-09 Conv_1_aggregated_features[4,1]
   + 6.8068e-11 Conv_1_aggregated_features[4,2]
   + 2.87298e-07 Conv_1_aggregated_features[4,4]
   + 1.28094e-07 Conv_1_aggregated_features[4,5]
   + 3.03159e-11 Conv_1_aggregated_features[4,6]
   + 2.96378e-05 Conv_1_aggregated_features[4,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[4,9]
   + 7.60786e-11 Conv_1_aggregated_features[4,10]
   + 9.73847e-08 Conv_1_aggregated_features[4,11]
   + 7.54748e-09 Conv_1_aggregated_features[4,12]
   + 1.58189e-09 Conv_1_aggregated_features[4,13] + Conv_1_t[4,2]
   = -1.53413e-08
 Conv_1_output_constraint[4,3]: 9.55069e-08 Conv_0_Relu_ts[4,3]
   - 4.06432e-09 Conv_0_Relu_ts[4,5] - 2.23188e-09 Conv_0_Relu_ts[4,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[4,12]
   + 4.18428e-06 Conv_0_Relu_ts[4,13]
   + 7.14864e-07 Conv_1_aggregated_features[4,3]
   - 9.90425e-11 Conv_1_aggregated_features[4,5]
   + 2.85992e-07 Conv_1_aggregated_features[4,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[4,12]
   - 7.41084e-08 Conv_1_aggregated_features[4,13] + Conv_1_t[4,3]
   = -3.15407e-06
 Conv_1_output_constraint[4,4]: 2.97121e-10 Conv_0_Relu_ts[4,0]
   + 9.07965e-08 Conv_0_Relu_ts[4,1] + 1.13683e-08 Conv_0_Relu_ts[4,2]
   + 7.68037e-08 Conv_0_Relu_ts[4,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[4,5]
   - 5.39198e-07 Conv_0_Relu_ts[4,6] - 6.18604e-12 Conv_0_Relu_ts[4,8]
   + 3.00799e-08 Conv_0_Relu_ts[4,10] - 1.66347e-05 Conv_0_Relu_ts[4,11]
   + 2.56472e-07 Conv_0_Relu_ts[4,12] - 3.27559e-05 Conv_0_Relu_ts[4,13]
   - 6.3717e-09 Conv_1_aggregated_features[4,0]
   + 3.53659e-08 Conv_1_aggregated_features[4,1]
   + 7.41964e-09 Conv_1_aggregated_features[4,2]
   + 0.00146958860568702 Conv_1_aggregated_features[4,4]
   + 1.41886e-06 Conv_1_aggregated_features[4,5]
   - 1.75898e-05 Conv_1_aggregated_features[4,6]
   + 0.0017772619612515 Conv_1_aggregated_features[4,8]
   + 7.64123e-10 Conv_1_aggregated_features[4,9]
   + 2.4839e-08 Conv_1_aggregated_features[4,10]
   - 2.68593e-07 Conv_1_aggregated_features[4,11]
   + 2.04143e-07 Conv_1_aggregated_features[4,12]
   + 4.75765e-08 Conv_1_aggregated_features[4,13] + Conv_1_t[4,4]
   = -8.58295e-08
 Conv_1_output_constraint[4,5]: - 1.12261e-07 Conv_0_Relu_ts[4,0]
   - 6.16837e-11 Conv_0_Relu_ts[4,1] - 1.43211e-10 Conv_0_Relu_ts[4,2]
   + 9.76965e-10 Conv_0_Relu_ts[4,3] - 8.36344e-05 Conv_0_Relu_ts[4,4]
   + 2.12422e-08 Conv_0_Relu_ts[4,5] - 2.04737e-07 Conv_0_Relu_ts[4,6]
   - 6.84967e-06 Conv_0_Relu_ts[4,8] + 2.60175e-05 Conv_0_Relu_ts[4,9]
   + 1.5048e-10 Conv_0_Relu_ts[4,10] - 2.26739e-07 Conv_0_Relu_ts[4,11]
   + 2.04031e-09 Conv_0_Relu_ts[4,12] + 3.64928e-12 Conv_0_Relu_ts[4,13]
   - 4.59272e-08 Conv_0_Relu_ts[4,15]
   + 8.09056e-08 Conv_1_aggregated_features[4,0]
   + 1.22547e-10 Conv_1_aggregated_features[4,1]
   + 1.29284e-10 Conv_1_aggregated_features[4,2]
   + 5.76885e-11 Conv_1_aggregated_features[4,3]
   + 5.76747e-08 Conv_1_aggregated_features[4,4]
   + 1.49451e-09 Conv_1_aggregated_features[4,5]
   + 9.85149e-10 Conv_1_aggregated_features[4,8]
   + 1.88005e-11 Conv_1_aggregated_features[4,9]
   + 3.09409e-11 Conv_1_aggregated_features[4,10]
   + 1.45468e-07 Conv_1_aggregated_features[4,11]
   + 2.76313e-09 Conv_1_aggregated_features[4,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[4,13]
   + Conv_1_t[4,5] = -4.66972e-09
 Conv_1_output_constraint[4,6]: 1.68259e-08 Conv_0_Relu_ts[4,0]
   + 2.66449e-06 Conv_0_Relu_ts[4,1] - 3.07804e-06 Conv_0_Relu_ts[4,2]
   - 1.56854e-10 Conv_0_Relu_ts[4,3] + 1.61577e-06 Conv_0_Relu_ts[4,4]
   + 9.63312e-13 Conv_0_Relu_ts[4,5] + 8.94799e-11 Conv_0_Relu_ts[4,6]
   + 0.00156858959235251 Conv_0_Relu_ts[4,8]
   + 3.71009e-11 Conv_0_Relu_ts[4,9] - 1.18851e-06 Conv_0_Relu_ts[4,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[4,11]
   + 2.08067e-05 Conv_0_Relu_ts[4,12] - 1.93794e-06 Conv_0_Relu_ts[4,13]
   + 2.00933e-08 Conv_1_aggregated_features[4,0]
   + 1.84936e-05 Conv_1_aggregated_features[4,1]
   - 2.40274e-06 Conv_1_aggregated_features[4,2]
   + 4.16152e-12 Conv_1_aggregated_features[4,3]
   - 9.5418e-06 Conv_1_aggregated_features[4,4]
   + 2.61288e-12 Conv_1_aggregated_features[4,5]
   - 5.0478e-05 Conv_1_aggregated_features[4,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[4,8]
   - 4.57774e-07 Conv_1_aggregated_features[4,9]
   - 7.00605e-07 Conv_1_aggregated_features[4,10]
   + 2.2587e-10 Conv_1_aggregated_features[4,11]
   + 4.15987e-05 Conv_1_aggregated_features[4,12]
   - 1.80554e-09 Conv_1_aggregated_features[4,13] + Conv_1_t[4,6]
   = -9.69776e-05
 Conv_1_output_constraint[4,7]: - 1.58234e-12 Conv_0_Relu_ts[4,0]
   + 6.43182e-10 Conv_0_Relu_ts[4,1] - 1.24733e-10 Conv_0_Relu_ts[4,2]
   + 7.71679e-06 Conv_0_Relu_ts[4,4] + 2.78803e-06 Conv_0_Relu_ts[4,6]
   + 2.03373e-06 Conv_0_Relu_ts[4,8] - 2.93243e-11 Conv_0_Relu_ts[4,9]
   - 2.23517e-10 Conv_0_Relu_ts[4,10] + 2.01832e-09 Conv_0_Relu_ts[4,12]
   + 5.32355e-07 Conv_1_aggregated_features[4,0]
   + 8.4797e-09 Conv_1_aggregated_features[4,1]
   - 1.50954e-10 Conv_1_aggregated_features[4,2]
   + 1.6387e-09 Conv_1_aggregated_features[4,4]
   + 2.45145e-06 Conv_1_aggregated_features[4,6]
   + 1.09862e-10 Conv_1_aggregated_features[4,7]
   - 1.68113e-05 Conv_1_aggregated_features[4,8]
   + 1.4449e-10 Conv_1_aggregated_features[4,9]
   - 4.62632e-10 Conv_1_aggregated_features[4,10]
   + 1.80231e-09 Conv_1_aggregated_features[4,12] + Conv_1_t[4,7]
   = -6.52043e-09
 Conv_1_output_constraint[4,8]: 2.0398e-13 Conv_1_aggregated_features[4,12]
   + Conv_1_t[4,8] = 2.40312e-18
 Conv_1_output_constraint[4,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[4,0]
   + 0.4666179716587067 Conv_0_Relu_ts[4,1]
   - 0.212794229388237 Conv_0_Relu_ts[4,2]
   + 1.31709e-13 Conv_0_Relu_ts[4,3] + 6.41139e-10 Conv_0_Relu_ts[4,4]
   - 1.36516e-10 Conv_0_Relu_ts[4,5]
   + 0.00123115978203714 Conv_0_Relu_ts[4,6]
   - 3.8707e-07 Conv_0_Relu_ts[4,7] - 9.99069e-10 Conv_0_Relu_ts[4,8]
   - 3.90229e-11 Conv_0_Relu_ts[4,9]
   - 0.3983311057090759 Conv_0_Relu_ts[4,10]
   + 1.4696e-09 Conv_0_Relu_ts[4,11]
   + 0.135991245508194 Conv_0_Relu_ts[4,12]
   - 5.03958e-07 Conv_0_Relu_ts[4,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[4,0]
   + 0.0555339716374874 Conv_1_aggregated_features[4,1]
   - 0.1152108833193779 Conv_1_aggregated_features[4,2]
   + 1.01018e-09 Conv_1_aggregated_features[4,4]
   - 2.65221e-10 Conv_1_aggregated_features[4,5]
   + 0.00134999 Conv_1_aggregated_features[4,6]
   - 7.7429e-07 Conv_1_aggregated_features[4,7]
   - 9.0928e-10 Conv_1_aggregated_features[4,8]
   - 4.02666e-11 Conv_1_aggregated_features[4,9]
   - 0.1311033517122269 Conv_1_aggregated_features[4,10]
   + 5.72141e-09 Conv_1_aggregated_features[4,11]
   - 1.062317252159119 Conv_1_aggregated_features[4,12]
   - 3.93908e-07 Conv_1_aggregated_features[4,13] + Conv_1_t[4,9]
   = 0.4778489172458649
 Conv_1_output_constraint[4,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[4,0]
   + 3.12754e-07 Conv_0_Relu_ts[4,1] + 1.78384e-08 Conv_0_Relu_ts[4,2]
   - 8.1105e-12 Conv_0_Relu_ts[4,3] - 1.21368e-06 Conv_0_Relu_ts[4,4]
   - 7.58218e-10 Conv_0_Relu_ts[4,5] + 6.37659e-05 Conv_0_Relu_ts[4,7]
   + 5.06994e-09 Conv_0_Relu_ts[4,8] + 4.38429e-08 Conv_0_Relu_ts[4,10]
   + 1.00038e-06 Conv_0_Relu_ts[4,12] - 1.00897e-07 Conv_0_Relu_ts[4,13]
   - 7.72898e-10 Conv_1_aggregated_features[4,0]
   + 1.93344e-07 Conv_1_aggregated_features[4,1]
   + 2.31957e-08 Conv_1_aggregated_features[4,2]
   - 5.28535e-07 Conv_1_aggregated_features[4,3]
   - 8.50617e-06 Conv_1_aggregated_features[4,4]
   - 1.86843e-09 Conv_1_aggregated_features[4,5]
   - 4.42334e-07 Conv_1_aggregated_features[4,8]
   - 1.66725e-07 Conv_1_aggregated_features[4,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[4,11]
   + 9.07032e-07 Conv_1_aggregated_features[4,12]
   + 1.19635e-11 Conv_1_aggregated_features[4,13] + Conv_1_t[4,10]
   = -7.05563e-07
 Conv_1_output_constraint[4,11]: 1.34578e-06 Conv_0_Relu_ts[4,0]
   + 2.25868e-05 Conv_0_Relu_ts[4,1] + 1.36528e-05 Conv_0_Relu_ts[4,2]
   + 6.61139e-05 Conv_0_Relu_ts[4,4] + 1.97151e-05 Conv_0_Relu_ts[4,5]
   - 2.52237e-09 Conv_0_Relu_ts[4,8] - 2.08745e-05 Conv_0_Relu_ts[4,9]
   + 4.27105e-05 Conv_0_Relu_ts[4,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[4,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[4,12]
   + 9.27492e-09 Conv_0_Relu_ts[4,13] - 1.59817e-08 Conv_0_Relu_ts[4,15]
   - 7.80059e-05 Conv_1_aggregated_features[4,0]
   + 3.38276e-05 Conv_1_aggregated_features[4,1]
   + 4.73819e-06 Conv_1_aggregated_features[4,2]
   - 0.00209861318580806 Conv_1_aggregated_features[4,4]
   - 4.14211e-05 Conv_1_aggregated_features[4,5]
   - 1.65027e-11 Conv_1_aggregated_features[4,8]
   + 2.09826e-05 Conv_1_aggregated_features[4,10]
   + 3.24145e-10 Conv_1_aggregated_features[4,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[4,12]
   - 6.08701e-11 Conv_1_aggregated_features[4,13] + Conv_1_t[4,11]
   = 3.07161e-05
 Conv_1_output_constraint[4,12]: 7.60796e-09 Conv_0_Relu_ts[4,1]
   + 2.00773e-10 Conv_0_Relu_ts[4,2] + 3.86788e-10 Conv_0_Relu_ts[4,3]
   + 3.1184e-11 Conv_0_Relu_ts[4,8] + 1.83806e-10 Conv_0_Relu_ts[4,10]
   + 2.13717e-08 Conv_0_Relu_ts[4,12] - 6.30213e-12 Conv_0_Relu_ts[4,13]
   + 2.90536e-09 Conv_1_aggregated_features[4,1]
   - 3.16398e-10 Conv_1_aggregated_features[4,2]
   - 2.02197e-09 Conv_1_aggregated_features[4,3]
   - 1.19877e-11 Conv_1_aggregated_features[4,8]
   - 5.09457e-10 Conv_1_aggregated_features[4,10]
   + 2.27248e-08 Conv_1_aggregated_features[4,12]
   - 1.24645e-13 Conv_1_aggregated_features[4,13] + Conv_1_t[4,12]
   = -3.75308e-09
 Conv_1_output_constraint[4,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[4,0]
   - 0.1570860594511032 Conv_0_Relu_ts[4,1]
   + 0.3520863950252533 Conv_0_Relu_ts[4,2]
   - 1.29036e-08 Conv_0_Relu_ts[4,4] + 3.83205e-07 Conv_0_Relu_ts[4,5]
   - 0.00537013821303844 Conv_0_Relu_ts[4,6]
   - 8.97018e-08 Conv_0_Relu_ts[4,7] + 1.712e-10 Conv_0_Relu_ts[4,8]
   + 1.81878e-10 Conv_0_Relu_ts[4,9]
   + 0.4303593337535858 Conv_0_Relu_ts[4,10]
   - 1.61554e-08 Conv_0_Relu_ts[4,11]
   - 0.9454745650291443 Conv_0_Relu_ts[4,12]
   + 4.84941e-06 Conv_0_Relu_ts[4,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[4,0]
   - 0.7010135054588318 Conv_1_aggregated_features[4,1]
   + 0.0964145809412003 Conv_1_aggregated_features[4,2]
   - 3.74041e-08 Conv_1_aggregated_features[4,4]
   + 5.47003e-07 Conv_1_aggregated_features[4,5]
   - 0.00514170713722706 Conv_1_aggregated_features[4,6]
   - 9.53448e-08 Conv_1_aggregated_features[4,7]
   - 2.07781e-10 Conv_1_aggregated_features[4,8]
   + 1.66638e-10 Conv_1_aggregated_features[4,9]
   + 0.1311885118484497 Conv_1_aggregated_features[4,10]
   - 9.23591e-08 Conv_1_aggregated_features[4,11]
   - 0.1851733475923538 Conv_1_aggregated_features[4,12]
   + 5.0438e-06 Conv_1_aggregated_features[4,13] + Conv_1_t[4,13]
   = -1.276697397232056
 Conv_1_output_constraint[4,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[4,0]
   + 0.00206982018426061 Conv_0_Relu_ts[4,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[4,2]
   - 9.77907e-13 Conv_0_Relu_ts[4,3] - 3.82624e-07 Conv_0_Relu_ts[4,4]
   + 1.19511e-07 Conv_0_Relu_ts[4,5] - 2.54839e-11 Conv_0_Relu_ts[4,6]
   + 4.99102e-07 Conv_0_Relu_ts[4,8] + 2.64257e-11 Conv_0_Relu_ts[4,9]
   - 0.00118294754065573 Conv_0_Relu_ts[4,10]
   + 4.8915e-08 Conv_0_Relu_ts[4,11]
   + 0.00699353078380227 Conv_0_Relu_ts[4,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[4,13]
   + 1.51033e-06 Conv_1_aggregated_features[4,0]
   + 0.00578453531488776 Conv_1_aggregated_features[4,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[4,2]
   + 5.14278e-09 Conv_1_aggregated_features[4,4]
   + 3.41007e-10 Conv_1_aggregated_features[4,5]
   + 1.47387e-08 Conv_1_aggregated_features[4,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[4,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[4,10]
   + 1.3873e-07 Conv_1_aggregated_features[4,11]
   + 0.0143724931403995 Conv_1_aggregated_features[4,12]
   - 6.50481e-05 Conv_1_aggregated_features[4,13] + Conv_1_t[4,14]
   = -0.00573839386925101
 Conv_1_output_constraint[4,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[4,0]
   + 3.59972e-09 Conv_0_Relu_ts[4,1] - 2.63862e-12 Conv_0_Relu_ts[4,2]
   - 0.00124379759654403 Conv_0_Relu_ts[4,4]
   + 1.90933e-08 Conv_0_Relu_ts[4,5] - 4.63531e-13 Conv_0_Relu_ts[4,6]
   - 1.54557e-05 Conv_0_Relu_ts[4,7] - 1.60225e-10 Conv_0_Relu_ts[4,8]
   + 2.49673e-06 Conv_0_Relu_ts[4,9] + 7.17346e-12 Conv_0_Relu_ts[4,10]
   - 3.53936e-11 Conv_0_Relu_ts[4,11] + 1.06168e-09 Conv_0_Relu_ts[4,12]
   - 4.87679e-07 Conv_0_Relu_ts[4,13] + 3.14464e-11 Conv_0_Relu_ts[4,15]
   - 0.000458086 Conv_1_aggregated_features[4,0]
   + 7.50942e-09 Conv_1_aggregated_features[4,1]
   - 3.13868e-12 Conv_1_aggregated_features[4,2]
   - 5.10815e-06 Conv_1_aggregated_features[4,4]
   - 1.93728e-06 Conv_1_aggregated_features[4,5]
   - 9.15832e-12 Conv_1_aggregated_features[4,6]
   + 5.1922e-09 Conv_1_aggregated_features[4,7]
   + 2.54628e-11 Conv_1_aggregated_features[4,8]
   - 1.10428e-11 Conv_1_aggregated_features[4,10]
   + 1.00597e-09 Conv_1_aggregated_features[4,12]
   + 7.86155e-08 Conv_1_aggregated_features[4,13] + Conv_1_t[4,15]
   = -1.04077e-08
 Conv_1_output_constraint[5,0]: 1.69823e-06 Conv_0_Relu_ts[5,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[5,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[5,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[5,12]
   + Conv_1_t[5,0] = -1.7536e-08
 Conv_1_output_constraint[5,1]: 6.64218e-05 Conv_0_Relu_ts[5,0]
   - 1.304573059082031 Conv_0_Relu_ts[5,1]
   - 0.00808643829077482 Conv_0_Relu_ts[5,2]
   + 1.10233e-12 Conv_0_Relu_ts[5,4] - 6.46578e-08 Conv_0_Relu_ts[5,6]
   + 3.59547e-13 Conv_0_Relu_ts[5,7] + 2.26512e-09 Conv_0_Relu_ts[5,8]
   - 1.48387e-10 Conv_0_Relu_ts[5,9]
   - 0.0375515259802341 Conv_0_Relu_ts[5,10]
   - 0.1779745519161224 Conv_0_Relu_ts[5,12]
   - 1.40572e-06 Conv_0_Relu_ts[5,13]
   + 4.89468e-05 Conv_1_aggregated_features[5,0]
   - 0.00799453444778919 Conv_1_aggregated_features[5,1]
   - 0.0869906693696976 Conv_1_aggregated_features[5,2]
   - 3.80301e-08 Conv_1_aggregated_features[5,6]
   + 2.46415e-09 Conv_1_aggregated_features[5,8]
   - 1.62219e-10 Conv_1_aggregated_features[5,9]
   - 0.2601335346698761 Conv_1_aggregated_features[5,10]
   + 0.4667477905750275 Conv_1_aggregated_features[5,12]
   - 1.71701e-06 Conv_1_aggregated_features[5,13] + Conv_1_t[5,1]
   = 0.3359648287296295
 Conv_1_output_constraint[5,2]: - 2.68276e-05 Conv_0_Relu_ts[5,0]
   + 7.4485e-11 Conv_0_Relu_ts[5,1] + 3.46068e-11 Conv_0_Relu_ts[5,2]
   - 5.94579e-10 Conv_0_Relu_ts[5,4] + 1.49975e-09 Conv_0_Relu_ts[5,5]
   + 4.00629e-09 Conv_0_Relu_ts[5,6] - 0.000385428 Conv_0_Relu_ts[5,8]
   - 9.65733e-05 Conv_0_Relu_ts[5,9] + 3.04566e-10 Conv_0_Relu_ts[5,10]
   + 2.67322e-05 Conv_0_Relu_ts[5,11] + 5.13886e-09 Conv_0_Relu_ts[5,12]
   - 6.79858e-13 Conv_0_Relu_ts[5,13]
   + 2.85913e-06 Conv_1_aggregated_features[5,0]
   + 3.02975e-09 Conv_1_aggregated_features[5,1]
   + 6.8068e-11 Conv_1_aggregated_features[5,2]
   + 2.87298e-07 Conv_1_aggregated_features[5,4]
   + 1.28094e-07 Conv_1_aggregated_features[5,5]
   + 3.03159e-11 Conv_1_aggregated_features[5,6]
   + 2.96378e-05 Conv_1_aggregated_features[5,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[5,9]
   + 7.60786e-11 Conv_1_aggregated_features[5,10]
   + 9.73847e-08 Conv_1_aggregated_features[5,11]
   + 7.54748e-09 Conv_1_aggregated_features[5,12]
   + 1.58189e-09 Conv_1_aggregated_features[5,13] + Conv_1_t[5,2]
   = -1.53413e-08
 Conv_1_output_constraint[5,3]: 9.55069e-08 Conv_0_Relu_ts[5,3]
   - 4.06432e-09 Conv_0_Relu_ts[5,5] - 2.23188e-09 Conv_0_Relu_ts[5,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[5,12]
   + 4.18428e-06 Conv_0_Relu_ts[5,13]
   + 7.14864e-07 Conv_1_aggregated_features[5,3]
   - 9.90425e-11 Conv_1_aggregated_features[5,5]
   + 2.85992e-07 Conv_1_aggregated_features[5,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[5,12]
   - 7.41084e-08 Conv_1_aggregated_features[5,13] + Conv_1_t[5,3]
   = -3.15407e-06
 Conv_1_output_constraint[5,4]: 2.97121e-10 Conv_0_Relu_ts[5,0]
   + 9.07965e-08 Conv_0_Relu_ts[5,1] + 1.13683e-08 Conv_0_Relu_ts[5,2]
   + 7.68037e-08 Conv_0_Relu_ts[5,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[5,5]
   - 5.39198e-07 Conv_0_Relu_ts[5,6] - 6.18604e-12 Conv_0_Relu_ts[5,8]
   + 3.00799e-08 Conv_0_Relu_ts[5,10] - 1.66347e-05 Conv_0_Relu_ts[5,11]
   + 2.56472e-07 Conv_0_Relu_ts[5,12] - 3.27559e-05 Conv_0_Relu_ts[5,13]
   - 6.3717e-09 Conv_1_aggregated_features[5,0]
   + 3.53659e-08 Conv_1_aggregated_features[5,1]
   + 7.41964e-09 Conv_1_aggregated_features[5,2]
   + 0.00146958860568702 Conv_1_aggregated_features[5,4]
   + 1.41886e-06 Conv_1_aggregated_features[5,5]
   - 1.75898e-05 Conv_1_aggregated_features[5,6]
   + 0.0017772619612515 Conv_1_aggregated_features[5,8]
   + 7.64123e-10 Conv_1_aggregated_features[5,9]
   + 2.4839e-08 Conv_1_aggregated_features[5,10]
   - 2.68593e-07 Conv_1_aggregated_features[5,11]
   + 2.04143e-07 Conv_1_aggregated_features[5,12]
   + 4.75765e-08 Conv_1_aggregated_features[5,13] + Conv_1_t[5,4]
   = -8.58295e-08
 Conv_1_output_constraint[5,5]: - 1.12261e-07 Conv_0_Relu_ts[5,0]
   - 6.16837e-11 Conv_0_Relu_ts[5,1] - 1.43211e-10 Conv_0_Relu_ts[5,2]
   + 9.76965e-10 Conv_0_Relu_ts[5,3] - 8.36344e-05 Conv_0_Relu_ts[5,4]
   + 2.12422e-08 Conv_0_Relu_ts[5,5] - 2.04737e-07 Conv_0_Relu_ts[5,6]
   - 6.84967e-06 Conv_0_Relu_ts[5,8] + 2.60175e-05 Conv_0_Relu_ts[5,9]
   + 1.5048e-10 Conv_0_Relu_ts[5,10] - 2.26739e-07 Conv_0_Relu_ts[5,11]
   + 2.04031e-09 Conv_0_Relu_ts[5,12] + 3.64928e-12 Conv_0_Relu_ts[5,13]
   - 4.59272e-08 Conv_0_Relu_ts[5,15]
   + 8.09056e-08 Conv_1_aggregated_features[5,0]
   + 1.22547e-10 Conv_1_aggregated_features[5,1]
   + 1.29284e-10 Conv_1_aggregated_features[5,2]
   + 5.76885e-11 Conv_1_aggregated_features[5,3]
   + 5.76747e-08 Conv_1_aggregated_features[5,4]
   + 1.49451e-09 Conv_1_aggregated_features[5,5]
   + 9.85149e-10 Conv_1_aggregated_features[5,8]
   + 1.88005e-11 Conv_1_aggregated_features[5,9]
   + 3.09409e-11 Conv_1_aggregated_features[5,10]
   + 1.45468e-07 Conv_1_aggregated_features[5,11]
   + 2.76313e-09 Conv_1_aggregated_features[5,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[5,13]
   + Conv_1_t[5,5] = -4.66972e-09
 Conv_1_output_constraint[5,6]: 1.68259e-08 Conv_0_Relu_ts[5,0]
   + 2.66449e-06 Conv_0_Relu_ts[5,1] - 3.07804e-06 Conv_0_Relu_ts[5,2]
   - 1.56854e-10 Conv_0_Relu_ts[5,3] + 1.61577e-06 Conv_0_Relu_ts[5,4]
   + 9.63312e-13 Conv_0_Relu_ts[5,5] + 8.94799e-11 Conv_0_Relu_ts[5,6]
   + 0.00156858959235251 Conv_0_Relu_ts[5,8]
   + 3.71009e-11 Conv_0_Relu_ts[5,9] - 1.18851e-06 Conv_0_Relu_ts[5,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[5,11]
   + 2.08067e-05 Conv_0_Relu_ts[5,12] - 1.93794e-06 Conv_0_Relu_ts[5,13]
   + 2.00933e-08 Conv_1_aggregated_features[5,0]
   + 1.84936e-05 Conv_1_aggregated_features[5,1]
   - 2.40274e-06 Conv_1_aggregated_features[5,2]
   + 4.16152e-12 Conv_1_aggregated_features[5,3]
   - 9.5418e-06 Conv_1_aggregated_features[5,4]
   + 2.61288e-12 Conv_1_aggregated_features[5,5]
   - 5.0478e-05 Conv_1_aggregated_features[5,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[5,8]
   - 4.57774e-07 Conv_1_aggregated_features[5,9]
   - 7.00605e-07 Conv_1_aggregated_features[5,10]
   + 2.2587e-10 Conv_1_aggregated_features[5,11]
   + 4.15987e-05 Conv_1_aggregated_features[5,12]
   - 1.80554e-09 Conv_1_aggregated_features[5,13] + Conv_1_t[5,6]
   = -9.69776e-05
 Conv_1_output_constraint[5,7]: - 1.58234e-12 Conv_0_Relu_ts[5,0]
   + 6.43182e-10 Conv_0_Relu_ts[5,1] - 1.24733e-10 Conv_0_Relu_ts[5,2]
   + 7.71679e-06 Conv_0_Relu_ts[5,4] + 2.78803e-06 Conv_0_Relu_ts[5,6]
   + 2.03373e-06 Conv_0_Relu_ts[5,8] - 2.93243e-11 Conv_0_Relu_ts[5,9]
   - 2.23517e-10 Conv_0_Relu_ts[5,10] + 2.01832e-09 Conv_0_Relu_ts[5,12]
   + 5.32355e-07 Conv_1_aggregated_features[5,0]
   + 8.4797e-09 Conv_1_aggregated_features[5,1]
   - 1.50954e-10 Conv_1_aggregated_features[5,2]
   + 1.6387e-09 Conv_1_aggregated_features[5,4]
   + 2.45145e-06 Conv_1_aggregated_features[5,6]
   + 1.09862e-10 Conv_1_aggregated_features[5,7]
   - 1.68113e-05 Conv_1_aggregated_features[5,8]
   + 1.4449e-10 Conv_1_aggregated_features[5,9]
   - 4.62632e-10 Conv_1_aggregated_features[5,10]
   + 1.80231e-09 Conv_1_aggregated_features[5,12] + Conv_1_t[5,7]
   = -6.52043e-09
 Conv_1_output_constraint[5,8]: 2.0398e-13 Conv_1_aggregated_features[5,12]
   + Conv_1_t[5,8] = 2.40312e-18
 Conv_1_output_constraint[5,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[5,0]
   + 0.4666179716587067 Conv_0_Relu_ts[5,1]
   - 0.212794229388237 Conv_0_Relu_ts[5,2]
   + 1.31709e-13 Conv_0_Relu_ts[5,3] + 6.41139e-10 Conv_0_Relu_ts[5,4]
   - 1.36516e-10 Conv_0_Relu_ts[5,5]
   + 0.00123115978203714 Conv_0_Relu_ts[5,6]
   - 3.8707e-07 Conv_0_Relu_ts[5,7] - 9.99069e-10 Conv_0_Relu_ts[5,8]
   - 3.90229e-11 Conv_0_Relu_ts[5,9]
   - 0.3983311057090759 Conv_0_Relu_ts[5,10]
   + 1.4696e-09 Conv_0_Relu_ts[5,11]
   + 0.135991245508194 Conv_0_Relu_ts[5,12]
   - 5.03958e-07 Conv_0_Relu_ts[5,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[5,0]
   + 0.0555339716374874 Conv_1_aggregated_features[5,1]
   - 0.1152108833193779 Conv_1_aggregated_features[5,2]
   + 1.01018e-09 Conv_1_aggregated_features[5,4]
   - 2.65221e-10 Conv_1_aggregated_features[5,5]
   + 0.00134999 Conv_1_aggregated_features[5,6]
   - 7.7429e-07 Conv_1_aggregated_features[5,7]
   - 9.0928e-10 Conv_1_aggregated_features[5,8]
   - 4.02666e-11 Conv_1_aggregated_features[5,9]
   - 0.1311033517122269 Conv_1_aggregated_features[5,10]
   + 5.72141e-09 Conv_1_aggregated_features[5,11]
   - 1.062317252159119 Conv_1_aggregated_features[5,12]
   - 3.93908e-07 Conv_1_aggregated_features[5,13] + Conv_1_t[5,9]
   = 0.4778489172458649
 Conv_1_output_constraint[5,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[5,0]
   + 3.12754e-07 Conv_0_Relu_ts[5,1] + 1.78384e-08 Conv_0_Relu_ts[5,2]
   - 8.1105e-12 Conv_0_Relu_ts[5,3] - 1.21368e-06 Conv_0_Relu_ts[5,4]
   - 7.58218e-10 Conv_0_Relu_ts[5,5] + 6.37659e-05 Conv_0_Relu_ts[5,7]
   + 5.06994e-09 Conv_0_Relu_ts[5,8] + 4.38429e-08 Conv_0_Relu_ts[5,10]
   + 1.00038e-06 Conv_0_Relu_ts[5,12] - 1.00897e-07 Conv_0_Relu_ts[5,13]
   - 7.72898e-10 Conv_1_aggregated_features[5,0]
   + 1.93344e-07 Conv_1_aggregated_features[5,1]
   + 2.31957e-08 Conv_1_aggregated_features[5,2]
   - 5.28535e-07 Conv_1_aggregated_features[5,3]
   - 8.50617e-06 Conv_1_aggregated_features[5,4]
   - 1.86843e-09 Conv_1_aggregated_features[5,5]
   - 4.42334e-07 Conv_1_aggregated_features[5,8]
   - 1.66725e-07 Conv_1_aggregated_features[5,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[5,11]
   + 9.07032e-07 Conv_1_aggregated_features[5,12]
   + 1.19635e-11 Conv_1_aggregated_features[5,13] + Conv_1_t[5,10]
   = -7.05563e-07
 Conv_1_output_constraint[5,11]: 1.34578e-06 Conv_0_Relu_ts[5,0]
   + 2.25868e-05 Conv_0_Relu_ts[5,1] + 1.36528e-05 Conv_0_Relu_ts[5,2]
   + 6.61139e-05 Conv_0_Relu_ts[5,4] + 1.97151e-05 Conv_0_Relu_ts[5,5]
   - 2.52237e-09 Conv_0_Relu_ts[5,8] - 2.08745e-05 Conv_0_Relu_ts[5,9]
   + 4.27105e-05 Conv_0_Relu_ts[5,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[5,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[5,12]
   + 9.27492e-09 Conv_0_Relu_ts[5,13] - 1.59817e-08 Conv_0_Relu_ts[5,15]
   - 7.80059e-05 Conv_1_aggregated_features[5,0]
   + 3.38276e-05 Conv_1_aggregated_features[5,1]
   + 4.73819e-06 Conv_1_aggregated_features[5,2]
   - 0.00209861318580806 Conv_1_aggregated_features[5,4]
   - 4.14211e-05 Conv_1_aggregated_features[5,5]
   - 1.65027e-11 Conv_1_aggregated_features[5,8]
   + 2.09826e-05 Conv_1_aggregated_features[5,10]
   + 3.24145e-10 Conv_1_aggregated_features[5,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[5,12]
   - 6.08701e-11 Conv_1_aggregated_features[5,13] + Conv_1_t[5,11]
   = 3.07161e-05
 Conv_1_output_constraint[5,12]: 7.60796e-09 Conv_0_Relu_ts[5,1]
   + 2.00773e-10 Conv_0_Relu_ts[5,2] + 3.86788e-10 Conv_0_Relu_ts[5,3]
   + 3.1184e-11 Conv_0_Relu_ts[5,8] + 1.83806e-10 Conv_0_Relu_ts[5,10]
   + 2.13717e-08 Conv_0_Relu_ts[5,12] - 6.30213e-12 Conv_0_Relu_ts[5,13]
   + 2.90536e-09 Conv_1_aggregated_features[5,1]
   - 3.16398e-10 Conv_1_aggregated_features[5,2]
   - 2.02197e-09 Conv_1_aggregated_features[5,3]
   - 1.19877e-11 Conv_1_aggregated_features[5,8]
   - 5.09457e-10 Conv_1_aggregated_features[5,10]
   + 2.27248e-08 Conv_1_aggregated_features[5,12]
   - 1.24645e-13 Conv_1_aggregated_features[5,13] + Conv_1_t[5,12]
   = -3.75308e-09
 Conv_1_output_constraint[5,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[5,0]
   - 0.1570860594511032 Conv_0_Relu_ts[5,1]
   + 0.3520863950252533 Conv_0_Relu_ts[5,2]
   - 1.29036e-08 Conv_0_Relu_ts[5,4] + 3.83205e-07 Conv_0_Relu_ts[5,5]
   - 0.00537013821303844 Conv_0_Relu_ts[5,6]
   - 8.97018e-08 Conv_0_Relu_ts[5,7] + 1.712e-10 Conv_0_Relu_ts[5,8]
   + 1.81878e-10 Conv_0_Relu_ts[5,9]
   + 0.4303593337535858 Conv_0_Relu_ts[5,10]
   - 1.61554e-08 Conv_0_Relu_ts[5,11]
   - 0.9454745650291443 Conv_0_Relu_ts[5,12]
   + 4.84941e-06 Conv_0_Relu_ts[5,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[5,0]
   - 0.7010135054588318 Conv_1_aggregated_features[5,1]
   + 0.0964145809412003 Conv_1_aggregated_features[5,2]
   - 3.74041e-08 Conv_1_aggregated_features[5,4]
   + 5.47003e-07 Conv_1_aggregated_features[5,5]
   - 0.00514170713722706 Conv_1_aggregated_features[5,6]
   - 9.53448e-08 Conv_1_aggregated_features[5,7]
   - 2.07781e-10 Conv_1_aggregated_features[5,8]
   + 1.66638e-10 Conv_1_aggregated_features[5,9]
   + 0.1311885118484497 Conv_1_aggregated_features[5,10]
   - 9.23591e-08 Conv_1_aggregated_features[5,11]
   - 0.1851733475923538 Conv_1_aggregated_features[5,12]
   + 5.0438e-06 Conv_1_aggregated_features[5,13] + Conv_1_t[5,13]
   = -1.276697397232056
 Conv_1_output_constraint[5,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[5,0]
   + 0.00206982018426061 Conv_0_Relu_ts[5,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[5,2]
   - 9.77907e-13 Conv_0_Relu_ts[5,3] - 3.82624e-07 Conv_0_Relu_ts[5,4]
   + 1.19511e-07 Conv_0_Relu_ts[5,5] - 2.54839e-11 Conv_0_Relu_ts[5,6]
   + 4.99102e-07 Conv_0_Relu_ts[5,8] + 2.64257e-11 Conv_0_Relu_ts[5,9]
   - 0.00118294754065573 Conv_0_Relu_ts[5,10]
   + 4.8915e-08 Conv_0_Relu_ts[5,11]
   + 0.00699353078380227 Conv_0_Relu_ts[5,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[5,13]
   + 1.51033e-06 Conv_1_aggregated_features[5,0]
   + 0.00578453531488776 Conv_1_aggregated_features[5,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[5,2]
   + 5.14278e-09 Conv_1_aggregated_features[5,4]
   + 3.41007e-10 Conv_1_aggregated_features[5,5]
   + 1.47387e-08 Conv_1_aggregated_features[5,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[5,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[5,10]
   + 1.3873e-07 Conv_1_aggregated_features[5,11]
   + 0.0143724931403995 Conv_1_aggregated_features[5,12]
   - 6.50481e-05 Conv_1_aggregated_features[5,13] + Conv_1_t[5,14]
   = -0.00573839386925101
 Conv_1_output_constraint[5,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[5,0]
   + 3.59972e-09 Conv_0_Relu_ts[5,1] - 2.63862e-12 Conv_0_Relu_ts[5,2]
   - 0.00124379759654403 Conv_0_Relu_ts[5,4]
   + 1.90933e-08 Conv_0_Relu_ts[5,5] - 4.63531e-13 Conv_0_Relu_ts[5,6]
   - 1.54557e-05 Conv_0_Relu_ts[5,7] - 1.60225e-10 Conv_0_Relu_ts[5,8]
   + 2.49673e-06 Conv_0_Relu_ts[5,9] + 7.17346e-12 Conv_0_Relu_ts[5,10]
   - 3.53936e-11 Conv_0_Relu_ts[5,11] + 1.06168e-09 Conv_0_Relu_ts[5,12]
   - 4.87679e-07 Conv_0_Relu_ts[5,13] + 3.14464e-11 Conv_0_Relu_ts[5,15]
   - 0.000458086 Conv_1_aggregated_features[5,0]
   + 7.50942e-09 Conv_1_aggregated_features[5,1]
   - 3.13868e-12 Conv_1_aggregated_features[5,2]
   - 5.10815e-06 Conv_1_aggregated_features[5,4]
   - 1.93728e-06 Conv_1_aggregated_features[5,5]
   - 9.15832e-12 Conv_1_aggregated_features[5,6]
   + 5.1922e-09 Conv_1_aggregated_features[5,7]
   + 2.54628e-11 Conv_1_aggregated_features[5,8]
   - 1.10428e-11 Conv_1_aggregated_features[5,10]
   + 1.00597e-09 Conv_1_aggregated_features[5,12]
   + 7.86155e-08 Conv_1_aggregated_features[5,13] + Conv_1_t[5,15]
   = -1.04077e-08
 Conv_1_output_constraint[6,0]: 1.69823e-06 Conv_0_Relu_ts[6,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[6,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[6,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[6,12]
   + Conv_1_t[6,0] = -1.7536e-08
 Conv_1_output_constraint[6,1]: 6.64218e-05 Conv_0_Relu_ts[6,0]
   - 1.304573059082031 Conv_0_Relu_ts[6,1]
   - 0.00808643829077482 Conv_0_Relu_ts[6,2]
   + 1.10233e-12 Conv_0_Relu_ts[6,4] - 6.46578e-08 Conv_0_Relu_ts[6,6]
   + 3.59547e-13 Conv_0_Relu_ts[6,7] + 2.26512e-09 Conv_0_Relu_ts[6,8]
   - 1.48387e-10 Conv_0_Relu_ts[6,9]
   - 0.0375515259802341 Conv_0_Relu_ts[6,10]
   - 0.1779745519161224 Conv_0_Relu_ts[6,12]
   - 1.40572e-06 Conv_0_Relu_ts[6,13]
   + 4.89468e-05 Conv_1_aggregated_features[6,0]
   - 0.00799453444778919 Conv_1_aggregated_features[6,1]
   - 0.0869906693696976 Conv_1_aggregated_features[6,2]
   - 3.80301e-08 Conv_1_aggregated_features[6,6]
   + 2.46415e-09 Conv_1_aggregated_features[6,8]
   - 1.62219e-10 Conv_1_aggregated_features[6,9]
   - 0.2601335346698761 Conv_1_aggregated_features[6,10]
   + 0.4667477905750275 Conv_1_aggregated_features[6,12]
   - 1.71701e-06 Conv_1_aggregated_features[6,13] + Conv_1_t[6,1]
   = 0.3359648287296295
 Conv_1_output_constraint[6,2]: - 2.68276e-05 Conv_0_Relu_ts[6,0]
   + 7.4485e-11 Conv_0_Relu_ts[6,1] + 3.46068e-11 Conv_0_Relu_ts[6,2]
   - 5.94579e-10 Conv_0_Relu_ts[6,4] + 1.49975e-09 Conv_0_Relu_ts[6,5]
   + 4.00629e-09 Conv_0_Relu_ts[6,6] - 0.000385428 Conv_0_Relu_ts[6,8]
   - 9.65733e-05 Conv_0_Relu_ts[6,9] + 3.04566e-10 Conv_0_Relu_ts[6,10]
   + 2.67322e-05 Conv_0_Relu_ts[6,11] + 5.13886e-09 Conv_0_Relu_ts[6,12]
   - 6.79858e-13 Conv_0_Relu_ts[6,13]
   + 2.85913e-06 Conv_1_aggregated_features[6,0]
   + 3.02975e-09 Conv_1_aggregated_features[6,1]
   + 6.8068e-11 Conv_1_aggregated_features[6,2]
   + 2.87298e-07 Conv_1_aggregated_features[6,4]
   + 1.28094e-07 Conv_1_aggregated_features[6,5]
   + 3.03159e-11 Conv_1_aggregated_features[6,6]
   + 2.96378e-05 Conv_1_aggregated_features[6,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[6,9]
   + 7.60786e-11 Conv_1_aggregated_features[6,10]
   + 9.73847e-08 Conv_1_aggregated_features[6,11]
   + 7.54748e-09 Conv_1_aggregated_features[6,12]
   + 1.58189e-09 Conv_1_aggregated_features[6,13] + Conv_1_t[6,2]
   = -1.53413e-08
 Conv_1_output_constraint[6,3]: 9.55069e-08 Conv_0_Relu_ts[6,3]
   - 4.06432e-09 Conv_0_Relu_ts[6,5] - 2.23188e-09 Conv_0_Relu_ts[6,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[6,12]
   + 4.18428e-06 Conv_0_Relu_ts[6,13]
   + 7.14864e-07 Conv_1_aggregated_features[6,3]
   - 9.90425e-11 Conv_1_aggregated_features[6,5]
   + 2.85992e-07 Conv_1_aggregated_features[6,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[6,12]
   - 7.41084e-08 Conv_1_aggregated_features[6,13] + Conv_1_t[6,3]
   = -3.15407e-06
 Conv_1_output_constraint[6,4]: 2.97121e-10 Conv_0_Relu_ts[6,0]
   + 9.07965e-08 Conv_0_Relu_ts[6,1] + 1.13683e-08 Conv_0_Relu_ts[6,2]
   + 7.68037e-08 Conv_0_Relu_ts[6,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[6,5]
   - 5.39198e-07 Conv_0_Relu_ts[6,6] - 6.18604e-12 Conv_0_Relu_ts[6,8]
   + 3.00799e-08 Conv_0_Relu_ts[6,10] - 1.66347e-05 Conv_0_Relu_ts[6,11]
   + 2.56472e-07 Conv_0_Relu_ts[6,12] - 3.27559e-05 Conv_0_Relu_ts[6,13]
   - 6.3717e-09 Conv_1_aggregated_features[6,0]
   + 3.53659e-08 Conv_1_aggregated_features[6,1]
   + 7.41964e-09 Conv_1_aggregated_features[6,2]
   + 0.00146958860568702 Conv_1_aggregated_features[6,4]
   + 1.41886e-06 Conv_1_aggregated_features[6,5]
   - 1.75898e-05 Conv_1_aggregated_features[6,6]
   + 0.0017772619612515 Conv_1_aggregated_features[6,8]
   + 7.64123e-10 Conv_1_aggregated_features[6,9]
   + 2.4839e-08 Conv_1_aggregated_features[6,10]
   - 2.68593e-07 Conv_1_aggregated_features[6,11]
   + 2.04143e-07 Conv_1_aggregated_features[6,12]
   + 4.75765e-08 Conv_1_aggregated_features[6,13] + Conv_1_t[6,4]
   = -8.58295e-08
 Conv_1_output_constraint[6,5]: - 1.12261e-07 Conv_0_Relu_ts[6,0]
   - 6.16837e-11 Conv_0_Relu_ts[6,1] - 1.43211e-10 Conv_0_Relu_ts[6,2]
   + 9.76965e-10 Conv_0_Relu_ts[6,3] - 8.36344e-05 Conv_0_Relu_ts[6,4]
   + 2.12422e-08 Conv_0_Relu_ts[6,5] - 2.04737e-07 Conv_0_Relu_ts[6,6]
   - 6.84967e-06 Conv_0_Relu_ts[6,8] + 2.60175e-05 Conv_0_Relu_ts[6,9]
   + 1.5048e-10 Conv_0_Relu_ts[6,10] - 2.26739e-07 Conv_0_Relu_ts[6,11]
   + 2.04031e-09 Conv_0_Relu_ts[6,12] + 3.64928e-12 Conv_0_Relu_ts[6,13]
   - 4.59272e-08 Conv_0_Relu_ts[6,15]
   + 8.09056e-08 Conv_1_aggregated_features[6,0]
   + 1.22547e-10 Conv_1_aggregated_features[6,1]
   + 1.29284e-10 Conv_1_aggregated_features[6,2]
   + 5.76885e-11 Conv_1_aggregated_features[6,3]
   + 5.76747e-08 Conv_1_aggregated_features[6,4]
   + 1.49451e-09 Conv_1_aggregated_features[6,5]
   + 9.85149e-10 Conv_1_aggregated_features[6,8]
   + 1.88005e-11 Conv_1_aggregated_features[6,9]
   + 3.09409e-11 Conv_1_aggregated_features[6,10]
   + 1.45468e-07 Conv_1_aggregated_features[6,11]
   + 2.76313e-09 Conv_1_aggregated_features[6,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[6,13]
   + Conv_1_t[6,5] = -4.66972e-09
 Conv_1_output_constraint[6,6]: 1.68259e-08 Conv_0_Relu_ts[6,0]
   + 2.66449e-06 Conv_0_Relu_ts[6,1] - 3.07804e-06 Conv_0_Relu_ts[6,2]
   - 1.56854e-10 Conv_0_Relu_ts[6,3] + 1.61577e-06 Conv_0_Relu_ts[6,4]
   + 9.63312e-13 Conv_0_Relu_ts[6,5] + 8.94799e-11 Conv_0_Relu_ts[6,6]
   + 0.00156858959235251 Conv_0_Relu_ts[6,8]
   + 3.71009e-11 Conv_0_Relu_ts[6,9] - 1.18851e-06 Conv_0_Relu_ts[6,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[6,11]
   + 2.08067e-05 Conv_0_Relu_ts[6,12] - 1.93794e-06 Conv_0_Relu_ts[6,13]
   + 2.00933e-08 Conv_1_aggregated_features[6,0]
   + 1.84936e-05 Conv_1_aggregated_features[6,1]
   - 2.40274e-06 Conv_1_aggregated_features[6,2]
   + 4.16152e-12 Conv_1_aggregated_features[6,3]
   - 9.5418e-06 Conv_1_aggregated_features[6,4]
   + 2.61288e-12 Conv_1_aggregated_features[6,5]
   - 5.0478e-05 Conv_1_aggregated_features[6,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[6,8]
   - 4.57774e-07 Conv_1_aggregated_features[6,9]
   - 7.00605e-07 Conv_1_aggregated_features[6,10]
   + 2.2587e-10 Conv_1_aggregated_features[6,11]
   + 4.15987e-05 Conv_1_aggregated_features[6,12]
   - 1.80554e-09 Conv_1_aggregated_features[6,13] + Conv_1_t[6,6]
   = -9.69776e-05
 Conv_1_output_constraint[6,7]: - 1.58234e-12 Conv_0_Relu_ts[6,0]
   + 6.43182e-10 Conv_0_Relu_ts[6,1] - 1.24733e-10 Conv_0_Relu_ts[6,2]
   + 7.71679e-06 Conv_0_Relu_ts[6,4] + 2.78803e-06 Conv_0_Relu_ts[6,6]
   + 2.03373e-06 Conv_0_Relu_ts[6,8] - 2.93243e-11 Conv_0_Relu_ts[6,9]
   - 2.23517e-10 Conv_0_Relu_ts[6,10] + 2.01832e-09 Conv_0_Relu_ts[6,12]
   + 5.32355e-07 Conv_1_aggregated_features[6,0]
   + 8.4797e-09 Conv_1_aggregated_features[6,1]
   - 1.50954e-10 Conv_1_aggregated_features[6,2]
   + 1.6387e-09 Conv_1_aggregated_features[6,4]
   + 2.45145e-06 Conv_1_aggregated_features[6,6]
   + 1.09862e-10 Conv_1_aggregated_features[6,7]
   - 1.68113e-05 Conv_1_aggregated_features[6,8]
   + 1.4449e-10 Conv_1_aggregated_features[6,9]
   - 4.62632e-10 Conv_1_aggregated_features[6,10]
   + 1.80231e-09 Conv_1_aggregated_features[6,12] + Conv_1_t[6,7]
   = -6.52043e-09
 Conv_1_output_constraint[6,8]: 2.0398e-13 Conv_1_aggregated_features[6,12]
   + Conv_1_t[6,8] = 2.40312e-18
 Conv_1_output_constraint[6,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[6,0]
   + 0.4666179716587067 Conv_0_Relu_ts[6,1]
   - 0.212794229388237 Conv_0_Relu_ts[6,2]
   + 1.31709e-13 Conv_0_Relu_ts[6,3] + 6.41139e-10 Conv_0_Relu_ts[6,4]
   - 1.36516e-10 Conv_0_Relu_ts[6,5]
   + 0.00123115978203714 Conv_0_Relu_ts[6,6]
   - 3.8707e-07 Conv_0_Relu_ts[6,7] - 9.99069e-10 Conv_0_Relu_ts[6,8]
   - 3.90229e-11 Conv_0_Relu_ts[6,9]
   - 0.3983311057090759 Conv_0_Relu_ts[6,10]
   + 1.4696e-09 Conv_0_Relu_ts[6,11]
   + 0.135991245508194 Conv_0_Relu_ts[6,12]
   - 5.03958e-07 Conv_0_Relu_ts[6,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[6,0]
   + 0.0555339716374874 Conv_1_aggregated_features[6,1]
   - 0.1152108833193779 Conv_1_aggregated_features[6,2]
   + 1.01018e-09 Conv_1_aggregated_features[6,4]
   - 2.65221e-10 Conv_1_aggregated_features[6,5]
   + 0.00134999 Conv_1_aggregated_features[6,6]
   - 7.7429e-07 Conv_1_aggregated_features[6,7]
   - 9.0928e-10 Conv_1_aggregated_features[6,8]
   - 4.02666e-11 Conv_1_aggregated_features[6,9]
   - 0.1311033517122269 Conv_1_aggregated_features[6,10]
   + 5.72141e-09 Conv_1_aggregated_features[6,11]
   - 1.062317252159119 Conv_1_aggregated_features[6,12]
   - 3.93908e-07 Conv_1_aggregated_features[6,13] + Conv_1_t[6,9]
   = 0.4778489172458649
 Conv_1_output_constraint[6,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[6,0]
   + 3.12754e-07 Conv_0_Relu_ts[6,1] + 1.78384e-08 Conv_0_Relu_ts[6,2]
   - 8.1105e-12 Conv_0_Relu_ts[6,3] - 1.21368e-06 Conv_0_Relu_ts[6,4]
   - 7.58218e-10 Conv_0_Relu_ts[6,5] + 6.37659e-05 Conv_0_Relu_ts[6,7]
   + 5.06994e-09 Conv_0_Relu_ts[6,8] + 4.38429e-08 Conv_0_Relu_ts[6,10]
   + 1.00038e-06 Conv_0_Relu_ts[6,12] - 1.00897e-07 Conv_0_Relu_ts[6,13]
   - 7.72898e-10 Conv_1_aggregated_features[6,0]
   + 1.93344e-07 Conv_1_aggregated_features[6,1]
   + 2.31957e-08 Conv_1_aggregated_features[6,2]
   - 5.28535e-07 Conv_1_aggregated_features[6,3]
   - 8.50617e-06 Conv_1_aggregated_features[6,4]
   - 1.86843e-09 Conv_1_aggregated_features[6,5]
   - 4.42334e-07 Conv_1_aggregated_features[6,8]
   - 1.66725e-07 Conv_1_aggregated_features[6,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[6,11]
   + 9.07032e-07 Conv_1_aggregated_features[6,12]
   + 1.19635e-11 Conv_1_aggregated_features[6,13] + Conv_1_t[6,10]
   = -7.05563e-07
 Conv_1_output_constraint[6,11]: 1.34578e-06 Conv_0_Relu_ts[6,0]
   + 2.25868e-05 Conv_0_Relu_ts[6,1] + 1.36528e-05 Conv_0_Relu_ts[6,2]
   + 6.61139e-05 Conv_0_Relu_ts[6,4] + 1.97151e-05 Conv_0_Relu_ts[6,5]
   - 2.52237e-09 Conv_0_Relu_ts[6,8] - 2.08745e-05 Conv_0_Relu_ts[6,9]
   + 4.27105e-05 Conv_0_Relu_ts[6,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[6,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[6,12]
   + 9.27492e-09 Conv_0_Relu_ts[6,13] - 1.59817e-08 Conv_0_Relu_ts[6,15]
   - 7.80059e-05 Conv_1_aggregated_features[6,0]
   + 3.38276e-05 Conv_1_aggregated_features[6,1]
   + 4.73819e-06 Conv_1_aggregated_features[6,2]
   - 0.00209861318580806 Conv_1_aggregated_features[6,4]
   - 4.14211e-05 Conv_1_aggregated_features[6,5]
   - 1.65027e-11 Conv_1_aggregated_features[6,8]
   + 2.09826e-05 Conv_1_aggregated_features[6,10]
   + 3.24145e-10 Conv_1_aggregated_features[6,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[6,12]
   - 6.08701e-11 Conv_1_aggregated_features[6,13] + Conv_1_t[6,11]
   = 3.07161e-05
 Conv_1_output_constraint[6,12]: 7.60796e-09 Conv_0_Relu_ts[6,1]
   + 2.00773e-10 Conv_0_Relu_ts[6,2] + 3.86788e-10 Conv_0_Relu_ts[6,3]
   + 3.1184e-11 Conv_0_Relu_ts[6,8] + 1.83806e-10 Conv_0_Relu_ts[6,10]
   + 2.13717e-08 Conv_0_Relu_ts[6,12] - 6.30213e-12 Conv_0_Relu_ts[6,13]
   + 2.90536e-09 Conv_1_aggregated_features[6,1]
   - 3.16398e-10 Conv_1_aggregated_features[6,2]
   - 2.02197e-09 Conv_1_aggregated_features[6,3]
   - 1.19877e-11 Conv_1_aggregated_features[6,8]
   - 5.09457e-10 Conv_1_aggregated_features[6,10]
   + 2.27248e-08 Conv_1_aggregated_features[6,12]
   - 1.24645e-13 Conv_1_aggregated_features[6,13] + Conv_1_t[6,12]
   = -3.75308e-09
 Conv_1_output_constraint[6,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[6,0]
   - 0.1570860594511032 Conv_0_Relu_ts[6,1]
   + 0.3520863950252533 Conv_0_Relu_ts[6,2]
   - 1.29036e-08 Conv_0_Relu_ts[6,4] + 3.83205e-07 Conv_0_Relu_ts[6,5]
   - 0.00537013821303844 Conv_0_Relu_ts[6,6]
   - 8.97018e-08 Conv_0_Relu_ts[6,7] + 1.712e-10 Conv_0_Relu_ts[6,8]
   + 1.81878e-10 Conv_0_Relu_ts[6,9]
   + 0.4303593337535858 Conv_0_Relu_ts[6,10]
   - 1.61554e-08 Conv_0_Relu_ts[6,11]
   - 0.9454745650291443 Conv_0_Relu_ts[6,12]
   + 4.84941e-06 Conv_0_Relu_ts[6,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[6,0]
   - 0.7010135054588318 Conv_1_aggregated_features[6,1]
   + 0.0964145809412003 Conv_1_aggregated_features[6,2]
   - 3.74041e-08 Conv_1_aggregated_features[6,4]
   + 5.47003e-07 Conv_1_aggregated_features[6,5]
   - 0.00514170713722706 Conv_1_aggregated_features[6,6]
   - 9.53448e-08 Conv_1_aggregated_features[6,7]
   - 2.07781e-10 Conv_1_aggregated_features[6,8]
   + 1.66638e-10 Conv_1_aggregated_features[6,9]
   + 0.1311885118484497 Conv_1_aggregated_features[6,10]
   - 9.23591e-08 Conv_1_aggregated_features[6,11]
   - 0.1851733475923538 Conv_1_aggregated_features[6,12]
   + 5.0438e-06 Conv_1_aggregated_features[6,13] + Conv_1_t[6,13]
   = -1.276697397232056
 Conv_1_output_constraint[6,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[6,0]
   + 0.00206982018426061 Conv_0_Relu_ts[6,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[6,2]
   - 9.77907e-13 Conv_0_Relu_ts[6,3] - 3.82624e-07 Conv_0_Relu_ts[6,4]
   + 1.19511e-07 Conv_0_Relu_ts[6,5] - 2.54839e-11 Conv_0_Relu_ts[6,6]
   + 4.99102e-07 Conv_0_Relu_ts[6,8] + 2.64257e-11 Conv_0_Relu_ts[6,9]
   - 0.00118294754065573 Conv_0_Relu_ts[6,10]
   + 4.8915e-08 Conv_0_Relu_ts[6,11]
   + 0.00699353078380227 Conv_0_Relu_ts[6,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[6,13]
   + 1.51033e-06 Conv_1_aggregated_features[6,0]
   + 0.00578453531488776 Conv_1_aggregated_features[6,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[6,2]
   + 5.14278e-09 Conv_1_aggregated_features[6,4]
   + 3.41007e-10 Conv_1_aggregated_features[6,5]
   + 1.47387e-08 Conv_1_aggregated_features[6,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[6,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[6,10]
   + 1.3873e-07 Conv_1_aggregated_features[6,11]
   + 0.0143724931403995 Conv_1_aggregated_features[6,12]
   - 6.50481e-05 Conv_1_aggregated_features[6,13] + Conv_1_t[6,14]
   = -0.00573839386925101
 Conv_1_output_constraint[6,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[6,0]
   + 3.59972e-09 Conv_0_Relu_ts[6,1] - 2.63862e-12 Conv_0_Relu_ts[6,2]
   - 0.00124379759654403 Conv_0_Relu_ts[6,4]
   + 1.90933e-08 Conv_0_Relu_ts[6,5] - 4.63531e-13 Conv_0_Relu_ts[6,6]
   - 1.54557e-05 Conv_0_Relu_ts[6,7] - 1.60225e-10 Conv_0_Relu_ts[6,8]
   + 2.49673e-06 Conv_0_Relu_ts[6,9] + 7.17346e-12 Conv_0_Relu_ts[6,10]
   - 3.53936e-11 Conv_0_Relu_ts[6,11] + 1.06168e-09 Conv_0_Relu_ts[6,12]
   - 4.87679e-07 Conv_0_Relu_ts[6,13] + 3.14464e-11 Conv_0_Relu_ts[6,15]
   - 0.000458086 Conv_1_aggregated_features[6,0]
   + 7.50942e-09 Conv_1_aggregated_features[6,1]
   - 3.13868e-12 Conv_1_aggregated_features[6,2]
   - 5.10815e-06 Conv_1_aggregated_features[6,4]
   - 1.93728e-06 Conv_1_aggregated_features[6,5]
   - 9.15832e-12 Conv_1_aggregated_features[6,6]
   + 5.1922e-09 Conv_1_aggregated_features[6,7]
   + 2.54628e-11 Conv_1_aggregated_features[6,8]
   - 1.10428e-11 Conv_1_aggregated_features[6,10]
   + 1.00597e-09 Conv_1_aggregated_features[6,12]
   + 7.86155e-08 Conv_1_aggregated_features[6,13] + Conv_1_t[6,15]
   = -1.04077e-08
 Conv_1_output_constraint[7,0]: 1.69823e-06 Conv_0_Relu_ts[7,8]
   + 4.3200247455388308e-04 Conv_0_Relu_ts[7,12]
   - 7.9638778697699308e-04 Conv_1_aggregated_features[7,8]
   + 7.4915686855092645e-04 Conv_1_aggregated_features[7,12]
   + Conv_1_t[7,0] = -1.7536e-08
 Conv_1_output_constraint[7,1]: 6.64218e-05 Conv_0_Relu_ts[7,0]
   - 1.304573059082031 Conv_0_Relu_ts[7,1]
   - 0.00808643829077482 Conv_0_Relu_ts[7,2]
   + 1.10233e-12 Conv_0_Relu_ts[7,4] - 6.46578e-08 Conv_0_Relu_ts[7,6]
   + 3.59547e-13 Conv_0_Relu_ts[7,7] + 2.26512e-09 Conv_0_Relu_ts[7,8]
   - 1.48387e-10 Conv_0_Relu_ts[7,9]
   - 0.0375515259802341 Conv_0_Relu_ts[7,10]
   - 0.1779745519161224 Conv_0_Relu_ts[7,12]
   - 1.40572e-06 Conv_0_Relu_ts[7,13]
   + 4.89468e-05 Conv_1_aggregated_features[7,0]
   - 0.00799453444778919 Conv_1_aggregated_features[7,1]
   - 0.0869906693696976 Conv_1_aggregated_features[7,2]
   - 3.80301e-08 Conv_1_aggregated_features[7,6]
   + 2.46415e-09 Conv_1_aggregated_features[7,8]
   - 1.62219e-10 Conv_1_aggregated_features[7,9]
   - 0.2601335346698761 Conv_1_aggregated_features[7,10]
   + 0.4667477905750275 Conv_1_aggregated_features[7,12]
   - 1.71701e-06 Conv_1_aggregated_features[7,13] + Conv_1_t[7,1]
   = 0.3359648287296295
 Conv_1_output_constraint[7,2]: - 2.68276e-05 Conv_0_Relu_ts[7,0]
   + 7.4485e-11 Conv_0_Relu_ts[7,1] + 3.46068e-11 Conv_0_Relu_ts[7,2]
   - 5.94579e-10 Conv_0_Relu_ts[7,4] + 1.49975e-09 Conv_0_Relu_ts[7,5]
   + 4.00629e-09 Conv_0_Relu_ts[7,6] - 0.000385428 Conv_0_Relu_ts[7,8]
   - 9.65733e-05 Conv_0_Relu_ts[7,9] + 3.04566e-10 Conv_0_Relu_ts[7,10]
   + 2.67322e-05 Conv_0_Relu_ts[7,11] + 5.13886e-09 Conv_0_Relu_ts[7,12]
   - 6.79858e-13 Conv_0_Relu_ts[7,13]
   + 2.85913e-06 Conv_1_aggregated_features[7,0]
   + 3.02975e-09 Conv_1_aggregated_features[7,1]
   + 6.8068e-11 Conv_1_aggregated_features[7,2]
   + 2.87298e-07 Conv_1_aggregated_features[7,4]
   + 1.28094e-07 Conv_1_aggregated_features[7,5]
   + 3.03159e-11 Conv_1_aggregated_features[7,6]
   + 2.96378e-05 Conv_1_aggregated_features[7,8]
   - 1.0900330380536616e-04 Conv_1_aggregated_features[7,9]
   + 7.60786e-11 Conv_1_aggregated_features[7,10]
   + 9.73847e-08 Conv_1_aggregated_features[7,11]
   + 7.54748e-09 Conv_1_aggregated_features[7,12]
   + 1.58189e-09 Conv_1_aggregated_features[7,13] + Conv_1_t[7,2]
   = -1.53413e-08
 Conv_1_output_constraint[7,3]: 9.55069e-08 Conv_0_Relu_ts[7,3]
   - 4.06432e-09 Conv_0_Relu_ts[7,5] - 2.23188e-09 Conv_0_Relu_ts[7,11]
   + 2.1080434089526534e-04 Conv_0_Relu_ts[7,12]
   + 4.18428e-06 Conv_0_Relu_ts[7,13]
   + 7.14864e-07 Conv_1_aggregated_features[7,3]
   - 9.90425e-11 Conv_1_aggregated_features[7,5]
   + 2.85992e-07 Conv_1_aggregated_features[7,11]
   + 5.0406914670020342e-04 Conv_1_aggregated_features[7,12]
   - 7.41084e-08 Conv_1_aggregated_features[7,13] + Conv_1_t[7,3]
   = -3.15407e-06
 Conv_1_output_constraint[7,4]: 2.97121e-10 Conv_0_Relu_ts[7,0]
   + 9.07965e-08 Conv_0_Relu_ts[7,1] + 1.13683e-08 Conv_0_Relu_ts[7,2]
   + 7.68037e-08 Conv_0_Relu_ts[7,4]
   - 2.3127558233682066e-04 Conv_0_Relu_ts[7,5]
   - 5.39198e-07 Conv_0_Relu_ts[7,6] - 6.18604e-12 Conv_0_Relu_ts[7,8]
   + 3.00799e-08 Conv_0_Relu_ts[7,10] - 1.66347e-05 Conv_0_Relu_ts[7,11]
   + 2.56472e-07 Conv_0_Relu_ts[7,12] - 3.27559e-05 Conv_0_Relu_ts[7,13]
   - 6.3717e-09 Conv_1_aggregated_features[7,0]
   + 3.53659e-08 Conv_1_aggregated_features[7,1]
   + 7.41964e-09 Conv_1_aggregated_features[7,2]
   + 0.00146958860568702 Conv_1_aggregated_features[7,4]
   + 1.41886e-06 Conv_1_aggregated_features[7,5]
   - 1.75898e-05 Conv_1_aggregated_features[7,6]
   + 0.0017772619612515 Conv_1_aggregated_features[7,8]
   + 7.64123e-10 Conv_1_aggregated_features[7,9]
   + 2.4839e-08 Conv_1_aggregated_features[7,10]
   - 2.68593e-07 Conv_1_aggregated_features[7,11]
   + 2.04143e-07 Conv_1_aggregated_features[7,12]
   + 4.75765e-08 Conv_1_aggregated_features[7,13] + Conv_1_t[7,4]
   = -8.58295e-08
 Conv_1_output_constraint[7,5]: - 1.12261e-07 Conv_0_Relu_ts[7,0]
   - 6.16837e-11 Conv_0_Relu_ts[7,1] - 1.43211e-10 Conv_0_Relu_ts[7,2]
   + 9.76965e-10 Conv_0_Relu_ts[7,3] - 8.36344e-05 Conv_0_Relu_ts[7,4]
   + 2.12422e-08 Conv_0_Relu_ts[7,5] - 2.04737e-07 Conv_0_Relu_ts[7,6]
   - 6.84967e-06 Conv_0_Relu_ts[7,8] + 2.60175e-05 Conv_0_Relu_ts[7,9]
   + 1.5048e-10 Conv_0_Relu_ts[7,10] - 2.26739e-07 Conv_0_Relu_ts[7,11]
   + 2.04031e-09 Conv_0_Relu_ts[7,12] + 3.64928e-12 Conv_0_Relu_ts[7,13]
   - 4.59272e-08 Conv_0_Relu_ts[7,15]
   + 8.09056e-08 Conv_1_aggregated_features[7,0]
   + 1.22547e-10 Conv_1_aggregated_features[7,1]
   + 1.29284e-10 Conv_1_aggregated_features[7,2]
   + 5.76885e-11 Conv_1_aggregated_features[7,3]
   + 5.76747e-08 Conv_1_aggregated_features[7,4]
   + 1.49451e-09 Conv_1_aggregated_features[7,5]
   + 9.85149e-10 Conv_1_aggregated_features[7,8]
   + 1.88005e-11 Conv_1_aggregated_features[7,9]
   + 3.09409e-11 Conv_1_aggregated_features[7,10]
   + 1.45468e-07 Conv_1_aggregated_features[7,11]
   + 2.76313e-09 Conv_1_aggregated_features[7,12]
   - 1.3591923925559968e-04 Conv_1_aggregated_features[7,13]
   + Conv_1_t[7,5] = -4.66972e-09
 Conv_1_output_constraint[7,6]: 1.68259e-08 Conv_0_Relu_ts[7,0]
   + 2.66449e-06 Conv_0_Relu_ts[7,1] - 3.07804e-06 Conv_0_Relu_ts[7,2]
   - 1.56854e-10 Conv_0_Relu_ts[7,3] + 1.61577e-06 Conv_0_Relu_ts[7,4]
   + 9.63312e-13 Conv_0_Relu_ts[7,5] + 8.94799e-11 Conv_0_Relu_ts[7,6]
   + 0.00156858959235251 Conv_0_Relu_ts[7,8]
   + 3.71009e-11 Conv_0_Relu_ts[7,9] - 1.18851e-06 Conv_0_Relu_ts[7,10]
   - 1.8981462926603854e-04 Conv_0_Relu_ts[7,11]
   + 2.08067e-05 Conv_0_Relu_ts[7,12] - 1.93794e-06 Conv_0_Relu_ts[7,13]
   + 2.00933e-08 Conv_1_aggregated_features[7,0]
   + 1.84936e-05 Conv_1_aggregated_features[7,1]
   - 2.40274e-06 Conv_1_aggregated_features[7,2]
   + 4.16152e-12 Conv_1_aggregated_features[7,3]
   - 9.5418e-06 Conv_1_aggregated_features[7,4]
   + 2.61288e-12 Conv_1_aggregated_features[7,5]
   - 5.0478e-05 Conv_1_aggregated_features[7,6]
   - 1.7703951743897051e-04 Conv_1_aggregated_features[7,8]
   - 4.57774e-07 Conv_1_aggregated_features[7,9]
   - 7.00605e-07 Conv_1_aggregated_features[7,10]
   + 2.2587e-10 Conv_1_aggregated_features[7,11]
   + 4.15987e-05 Conv_1_aggregated_features[7,12]
   - 1.80554e-09 Conv_1_aggregated_features[7,13] + Conv_1_t[7,6]
   = -9.69776e-05
 Conv_1_output_constraint[7,7]: - 1.58234e-12 Conv_0_Relu_ts[7,0]
   + 6.43182e-10 Conv_0_Relu_ts[7,1] - 1.24733e-10 Conv_0_Relu_ts[7,2]
   + 7.71679e-06 Conv_0_Relu_ts[7,4] + 2.78803e-06 Conv_0_Relu_ts[7,6]
   + 2.03373e-06 Conv_0_Relu_ts[7,8] - 2.93243e-11 Conv_0_Relu_ts[7,9]
   - 2.23517e-10 Conv_0_Relu_ts[7,10] + 2.01832e-09 Conv_0_Relu_ts[7,12]
   + 5.32355e-07 Conv_1_aggregated_features[7,0]
   + 8.4797e-09 Conv_1_aggregated_features[7,1]
   - 1.50954e-10 Conv_1_aggregated_features[7,2]
   + 1.6387e-09 Conv_1_aggregated_features[7,4]
   + 2.45145e-06 Conv_1_aggregated_features[7,6]
   + 1.09862e-10 Conv_1_aggregated_features[7,7]
   - 1.68113e-05 Conv_1_aggregated_features[7,8]
   + 1.4449e-10 Conv_1_aggregated_features[7,9]
   - 4.62632e-10 Conv_1_aggregated_features[7,10]
   + 1.80231e-09 Conv_1_aggregated_features[7,12] + Conv_1_t[7,7]
   = -6.52043e-09
 Conv_1_output_constraint[7,8]: 2.0398e-13 Conv_1_aggregated_features[7,12]
   + Conv_1_t[7,8] = 2.40312e-18
 Conv_1_output_constraint[7,9]:
   - 1.2523162877187133e-04 Conv_0_Relu_ts[7,0]
   + 0.4666179716587067 Conv_0_Relu_ts[7,1]
   - 0.212794229388237 Conv_0_Relu_ts[7,2]
   + 1.31709e-13 Conv_0_Relu_ts[7,3] + 6.41139e-10 Conv_0_Relu_ts[7,4]
   - 1.36516e-10 Conv_0_Relu_ts[7,5]
   + 0.00123115978203714 Conv_0_Relu_ts[7,6]
   - 3.8707e-07 Conv_0_Relu_ts[7,7] - 9.99069e-10 Conv_0_Relu_ts[7,8]
   - 3.90229e-11 Conv_0_Relu_ts[7,9]
   - 0.3983311057090759 Conv_0_Relu_ts[7,10]
   + 1.4696e-09 Conv_0_Relu_ts[7,11]
   + 0.135991245508194 Conv_0_Relu_ts[7,12]
   - 5.03958e-07 Conv_0_Relu_ts[7,13]
   - 1.4372733130585402e-04 Conv_1_aggregated_features[7,0]
   + 0.0555339716374874 Conv_1_aggregated_features[7,1]
   - 0.1152108833193779 Conv_1_aggregated_features[7,2]
   + 1.01018e-09 Conv_1_aggregated_features[7,4]
   - 2.65221e-10 Conv_1_aggregated_features[7,5]
   + 0.00134999 Conv_1_aggregated_features[7,6]
   - 7.7429e-07 Conv_1_aggregated_features[7,7]
   - 9.0928e-10 Conv_1_aggregated_features[7,8]
   - 4.02666e-11 Conv_1_aggregated_features[7,9]
   - 0.1311033517122269 Conv_1_aggregated_features[7,10]
   + 5.72141e-09 Conv_1_aggregated_features[7,11]
   - 1.062317252159119 Conv_1_aggregated_features[7,12]
   - 3.93908e-07 Conv_1_aggregated_features[7,13] + Conv_1_t[7,9]
   = 0.4778489172458649
 Conv_1_output_constraint[7,10]:
   - 2.3953753407113254e-04 Conv_0_Relu_ts[7,0]
   + 3.12754e-07 Conv_0_Relu_ts[7,1] + 1.78384e-08 Conv_0_Relu_ts[7,2]
   - 8.1105e-12 Conv_0_Relu_ts[7,3] - 1.21368e-06 Conv_0_Relu_ts[7,4]
   - 7.58218e-10 Conv_0_Relu_ts[7,5] + 6.37659e-05 Conv_0_Relu_ts[7,7]
   + 5.06994e-09 Conv_0_Relu_ts[7,8] + 4.38429e-08 Conv_0_Relu_ts[7,10]
   + 1.00038e-06 Conv_0_Relu_ts[7,12] - 1.00897e-07 Conv_0_Relu_ts[7,13]
   - 7.72898e-10 Conv_1_aggregated_features[7,0]
   + 1.93344e-07 Conv_1_aggregated_features[7,1]
   + 2.31957e-08 Conv_1_aggregated_features[7,2]
   - 5.28535e-07 Conv_1_aggregated_features[7,3]
   - 8.50617e-06 Conv_1_aggregated_features[7,4]
   - 1.86843e-09 Conv_1_aggregated_features[7,5]
   - 4.42334e-07 Conv_1_aggregated_features[7,8]
   - 1.66725e-07 Conv_1_aggregated_features[7,10]
   - 4.975832998752594e-04 Conv_1_aggregated_features[7,11]
   + 9.07032e-07 Conv_1_aggregated_features[7,12]
   + 1.19635e-11 Conv_1_aggregated_features[7,13] + Conv_1_t[7,10]
   = -7.05563e-07
 Conv_1_output_constraint[7,11]: 1.34578e-06 Conv_0_Relu_ts[7,0]
   + 2.25868e-05 Conv_0_Relu_ts[7,1] + 1.36528e-05 Conv_0_Relu_ts[7,2]
   + 6.61139e-05 Conv_0_Relu_ts[7,4] + 1.97151e-05 Conv_0_Relu_ts[7,5]
   - 2.52237e-09 Conv_0_Relu_ts[7,8] - 2.08745e-05 Conv_0_Relu_ts[7,9]
   + 4.27105e-05 Conv_0_Relu_ts[7,10]
   + 3.633717424236238e-04 Conv_0_Relu_ts[7,11]
   + 1.4846351405140013e-04 Conv_0_Relu_ts[7,12]
   + 9.27492e-09 Conv_0_Relu_ts[7,13] - 1.59817e-08 Conv_0_Relu_ts[7,15]
   - 7.80059e-05 Conv_1_aggregated_features[7,0]
   + 3.38276e-05 Conv_1_aggregated_features[7,1]
   + 4.73819e-06 Conv_1_aggregated_features[7,2]
   - 0.00209861318580806 Conv_1_aggregated_features[7,4]
   - 4.14211e-05 Conv_1_aggregated_features[7,5]
   - 1.65027e-11 Conv_1_aggregated_features[7,8]
   + 2.09826e-05 Conv_1_aggregated_features[7,10]
   + 3.24145e-10 Conv_1_aggregated_features[7,11]
   + 1.9054242875427008e-04 Conv_1_aggregated_features[7,12]
   - 6.08701e-11 Conv_1_aggregated_features[7,13] + Conv_1_t[7,11]
   = 3.07161e-05
 Conv_1_output_constraint[7,12]: 7.60796e-09 Conv_0_Relu_ts[7,1]
   + 2.00773e-10 Conv_0_Relu_ts[7,2] + 3.86788e-10 Conv_0_Relu_ts[7,3]
   + 3.1184e-11 Conv_0_Relu_ts[7,8] + 1.83806e-10 Conv_0_Relu_ts[7,10]
   + 2.13717e-08 Conv_0_Relu_ts[7,12] - 6.30213e-12 Conv_0_Relu_ts[7,13]
   + 2.90536e-09 Conv_1_aggregated_features[7,1]
   - 3.16398e-10 Conv_1_aggregated_features[7,2]
   - 2.02197e-09 Conv_1_aggregated_features[7,3]
   - 1.19877e-11 Conv_1_aggregated_features[7,8]
   - 5.09457e-10 Conv_1_aggregated_features[7,10]
   + 2.27248e-08 Conv_1_aggregated_features[7,12]
   - 1.24645e-13 Conv_1_aggregated_features[7,13] + Conv_1_t[7,12]
   = -3.75308e-09
 Conv_1_output_constraint[7,13]: 1.3043428771197796e-04 Conv_0_Relu_ts[7,0]
   - 0.1570860594511032 Conv_0_Relu_ts[7,1]
   + 0.3520863950252533 Conv_0_Relu_ts[7,2]
   - 1.29036e-08 Conv_0_Relu_ts[7,4] + 3.83205e-07 Conv_0_Relu_ts[7,5]
   - 0.00537013821303844 Conv_0_Relu_ts[7,6]
   - 8.97018e-08 Conv_0_Relu_ts[7,7] + 1.712e-10 Conv_0_Relu_ts[7,8]
   + 1.81878e-10 Conv_0_Relu_ts[7,9]
   + 0.4303593337535858 Conv_0_Relu_ts[7,10]
   - 1.61554e-08 Conv_0_Relu_ts[7,11]
   - 0.9454745650291443 Conv_0_Relu_ts[7,12]
   + 4.84941e-06 Conv_0_Relu_ts[7,13]
   + 1.7625710461288691e-04 Conv_1_aggregated_features[7,0]
   - 0.7010135054588318 Conv_1_aggregated_features[7,1]
   + 0.0964145809412003 Conv_1_aggregated_features[7,2]
   - 3.74041e-08 Conv_1_aggregated_features[7,4]
   + 5.47003e-07 Conv_1_aggregated_features[7,5]
   - 0.00514170713722706 Conv_1_aggregated_features[7,6]
   - 9.53448e-08 Conv_1_aggregated_features[7,7]
   - 2.07781e-10 Conv_1_aggregated_features[7,8]
   + 1.66638e-10 Conv_1_aggregated_features[7,9]
   + 0.1311885118484497 Conv_1_aggregated_features[7,10]
   - 9.23591e-08 Conv_1_aggregated_features[7,11]
   - 0.1851733475923538 Conv_1_aggregated_features[7,12]
   + 5.0438e-06 Conv_1_aggregated_features[7,13] + Conv_1_t[7,13]
   = -1.276697397232056
 Conv_1_output_constraint[7,14]: 4.6531413681805134e-04 Conv_0_Relu_ts[7,0]
   + 0.00206982018426061 Conv_0_Relu_ts[7,1]
   - 1.2881518341600895e-04 Conv_0_Relu_ts[7,2]
   - 9.77907e-13 Conv_0_Relu_ts[7,3] - 3.82624e-07 Conv_0_Relu_ts[7,4]
   + 1.19511e-07 Conv_0_Relu_ts[7,5] - 2.54839e-11 Conv_0_Relu_ts[7,6]
   + 4.99102e-07 Conv_0_Relu_ts[7,8] + 2.64257e-11 Conv_0_Relu_ts[7,9]
   - 0.00118294754065573 Conv_0_Relu_ts[7,10]
   + 4.8915e-08 Conv_0_Relu_ts[7,11]
   + 0.00699353078380227 Conv_0_Relu_ts[7,12]
   - 1.2067411444149911e-04 Conv_0_Relu_ts[7,13]
   + 1.51033e-06 Conv_1_aggregated_features[7,0]
   + 0.00578453531488776 Conv_1_aggregated_features[7,1]
   + 2.260124747408554e-04 Conv_1_aggregated_features[7,2]
   + 5.14278e-09 Conv_1_aggregated_features[7,4]
   + 3.41007e-10 Conv_1_aggregated_features[7,5]
   + 1.47387e-08 Conv_1_aggregated_features[7,6]
   - 1.2065647752024233e-04 Conv_1_aggregated_features[7,8]
   - 9.9785311613231897e-04 Conv_1_aggregated_features[7,10]
   + 1.3873e-07 Conv_1_aggregated_features[7,11]
   + 0.0143724931403995 Conv_1_aggregated_features[7,12]
   - 6.50481e-05 Conv_1_aggregated_features[7,13] + Conv_1_t[7,14]
   = -0.00573839386925101
 Conv_1_output_constraint[7,15]:
   - 3.2710150117054582e-04 Conv_0_Relu_ts[7,0]
   + 3.59972e-09 Conv_0_Relu_ts[7,1] - 2.63862e-12 Conv_0_Relu_ts[7,2]
   - 0.00124379759654403 Conv_0_Relu_ts[7,4]
   + 1.90933e-08 Conv_0_Relu_ts[7,5] - 4.63531e-13 Conv_0_Relu_ts[7,6]
   - 1.54557e-05 Conv_0_Relu_ts[7,7] - 1.60225e-10 Conv_0_Relu_ts[7,8]
   + 2.49673e-06 Conv_0_Relu_ts[7,9] + 7.17346e-12 Conv_0_Relu_ts[7,10]
   - 3.53936e-11 Conv_0_Relu_ts[7,11] + 1.06168e-09 Conv_0_Relu_ts[7,12]
   - 4.87679e-07 Conv_0_Relu_ts[7,13] + 3.14464e-11 Conv_0_Relu_ts[7,15]
   - 0.000458086 Conv_1_aggregated_features[7,0]
   + 7.50942e-09 Conv_1_aggregated_features[7,1]
   - 3.13868e-12 Conv_1_aggregated_features[7,2]
   - 5.10815e-06 Conv_1_aggregated_features[7,4]
   - 1.93728e-06 Conv_1_aggregated_features[7,5]
   - 9.15832e-12 Conv_1_aggregated_features[7,6]
   + 5.1922e-09 Conv_1_aggregated_features[7,7]
   + 2.54628e-11 Conv_1_aggregated_features[7,8]
   - 1.10428e-11 Conv_1_aggregated_features[7,10]
   + 1.00597e-09 Conv_1_aggregated_features[7,12]
   + 7.86155e-08 Conv_1_aggregated_features[7,13] + Conv_1_t[7,15]
   = -1.04077e-08
 Conv_1_Relu_constraint_1[0,0]: - Conv_1_t[0,0] + Conv_1_Relu_ts[0,0]
   - Conv_1_Relu_ss[0,0] = 0
 Conv_1_Relu_constraint_1[0,1]: - Conv_1_t[0,1] + Conv_1_Relu_ts[0,1]
   - Conv_1_Relu_ss[0,1] = 0
 Conv_1_Relu_constraint_1[0,2]: - Conv_1_t[0,2] + Conv_1_Relu_ts[0,2]
   - Conv_1_Relu_ss[0,2] = 0
 Conv_1_Relu_constraint_1[0,3]: - Conv_1_t[0,3] + Conv_1_Relu_ts[0,3]
   - Conv_1_Relu_ss[0,3] = 0
 Conv_1_Relu_constraint_1[0,4]: - Conv_1_t[0,4] + Conv_1_Relu_ts[0,4]
   - Conv_1_Relu_ss[0,4] = 0
 Conv_1_Relu_constraint_1[0,5]: - Conv_1_t[0,5] + Conv_1_Relu_ts[0,5]
   - Conv_1_Relu_ss[0,5] = 0
 Conv_1_Relu_constraint_1[0,6]: - Conv_1_t[0,6] + Conv_1_Relu_ts[0,6]
   - Conv_1_Relu_ss[0,6] = 0
 Conv_1_Relu_constraint_1[0,7]: - Conv_1_t[0,7] + Conv_1_Relu_ts[0,7]
   - Conv_1_Relu_ss[0,7] = 0
 Conv_1_Relu_constraint_1[0,8]: - Conv_1_t[0,8] + Conv_1_Relu_ts[0,8]
   - Conv_1_Relu_ss[0,8] = 0
 Conv_1_Relu_constraint_1[0,9]: - Conv_1_t[0,9] + Conv_1_Relu_ts[0,9]
   - Conv_1_Relu_ss[0,9] = 0
 Conv_1_Relu_constraint_1[0,10]: - Conv_1_t[0,10] + Conv_1_Relu_ts[0,10]
   - Conv_1_Relu_ss[0,10] = 0
 Conv_1_Relu_constraint_1[0,11]: - Conv_1_t[0,11] + Conv_1_Relu_ts[0,11]
   - Conv_1_Relu_ss[0,11] = 0
 Conv_1_Relu_constraint_1[0,12]: - Conv_1_t[0,12] + Conv_1_Relu_ts[0,12]
   - Conv_1_Relu_ss[0,12] = 0
 Conv_1_Relu_constraint_1[0,13]: - Conv_1_t[0,13] + Conv_1_Relu_ts[0,13]
   - Conv_1_Relu_ss[0,13] = 0
 Conv_1_Relu_constraint_1[0,14]: - Conv_1_t[0,14] + Conv_1_Relu_ts[0,14]
   - Conv_1_Relu_ss[0,14] = 0
 Conv_1_Relu_constraint_1[0,15]: - Conv_1_t[0,15] + Conv_1_Relu_ts[0,15]
   - Conv_1_Relu_ss[0,15] = 0
 Conv_1_Relu_constraint_1[1,0]: - Conv_1_t[1,0] + Conv_1_Relu_ts[1,0]
   - Conv_1_Relu_ss[1,0] = 0
 Conv_1_Relu_constraint_1[1,1]: - Conv_1_t[1,1] + Conv_1_Relu_ts[1,1]
   - Conv_1_Relu_ss[1,1] = 0
 Conv_1_Relu_constraint_1[1,2]: - Conv_1_t[1,2] + Conv_1_Relu_ts[1,2]
   - Conv_1_Relu_ss[1,2] = 0
 Conv_1_Relu_constraint_1[1,3]: - Conv_1_t[1,3] + Conv_1_Relu_ts[1,3]
   - Conv_1_Relu_ss[1,3] = 0
 Conv_1_Relu_constraint_1[1,4]: - Conv_1_t[1,4] + Conv_1_Relu_ts[1,4]
   - Conv_1_Relu_ss[1,4] = 0
 Conv_1_Relu_constraint_1[1,5]: - Conv_1_t[1,5] + Conv_1_Relu_ts[1,5]
   - Conv_1_Relu_ss[1,5] = 0
 Conv_1_Relu_constraint_1[1,6]: - Conv_1_t[1,6] + Conv_1_Relu_ts[1,6]
   - Conv_1_Relu_ss[1,6] = 0
 Conv_1_Relu_constraint_1[1,7]: - Conv_1_t[1,7] + Conv_1_Relu_ts[1,7]
   - Conv_1_Relu_ss[1,7] = 0
 Conv_1_Relu_constraint_1[1,8]: - Conv_1_t[1,8] + Conv_1_Relu_ts[1,8]
   - Conv_1_Relu_ss[1,8] = 0
 Conv_1_Relu_constraint_1[1,9]: - Conv_1_t[1,9] + Conv_1_Relu_ts[1,9]
   - Conv_1_Relu_ss[1,9] = 0
 Conv_1_Relu_constraint_1[1,10]: - Conv_1_t[1,10] + Conv_1_Relu_ts[1,10]
   - Conv_1_Relu_ss[1,10] = 0
 Conv_1_Relu_constraint_1[1,11]: - Conv_1_t[1,11] + Conv_1_Relu_ts[1,11]
   - Conv_1_Relu_ss[1,11] = 0
 Conv_1_Relu_constraint_1[1,12]: - Conv_1_t[1,12] + Conv_1_Relu_ts[1,12]
   - Conv_1_Relu_ss[1,12] = 0
 Conv_1_Relu_constraint_1[1,13]: - Conv_1_t[1,13] + Conv_1_Relu_ts[1,13]
   - Conv_1_Relu_ss[1,13] = 0
 Conv_1_Relu_constraint_1[1,14]: - Conv_1_t[1,14] + Conv_1_Relu_ts[1,14]
   - Conv_1_Relu_ss[1,14] = 0
 Conv_1_Relu_constraint_1[1,15]: - Conv_1_t[1,15] + Conv_1_Relu_ts[1,15]
   - Conv_1_Relu_ss[1,15] = 0
 Conv_1_Relu_constraint_1[2,0]: - Conv_1_t[2,0] + Conv_1_Relu_ts[2,0]
   - Conv_1_Relu_ss[2,0] = 0
 Conv_1_Relu_constraint_1[2,1]: - Conv_1_t[2,1] + Conv_1_Relu_ts[2,1]
   - Conv_1_Relu_ss[2,1] = 0
 Conv_1_Relu_constraint_1[2,2]: - Conv_1_t[2,2] + Conv_1_Relu_ts[2,2]
   - Conv_1_Relu_ss[2,2] = 0
 Conv_1_Relu_constraint_1[2,3]: - Conv_1_t[2,3] + Conv_1_Relu_ts[2,3]
   - Conv_1_Relu_ss[2,3] = 0
 Conv_1_Relu_constraint_1[2,4]: - Conv_1_t[2,4] + Conv_1_Relu_ts[2,4]
   - Conv_1_Relu_ss[2,4] = 0
 Conv_1_Relu_constraint_1[2,5]: - Conv_1_t[2,5] + Conv_1_Relu_ts[2,5]
   - Conv_1_Relu_ss[2,5] = 0
 Conv_1_Relu_constraint_1[2,6]: - Conv_1_t[2,6] + Conv_1_Relu_ts[2,6]
   - Conv_1_Relu_ss[2,6] = 0
 Conv_1_Relu_constraint_1[2,7]: - Conv_1_t[2,7] + Conv_1_Relu_ts[2,7]
   - Conv_1_Relu_ss[2,7] = 0
 Conv_1_Relu_constraint_1[2,8]: - Conv_1_t[2,8] + Conv_1_Relu_ts[2,8]
   - Conv_1_Relu_ss[2,8] = 0
 Conv_1_Relu_constraint_1[2,9]: - Conv_1_t[2,9] + Conv_1_Relu_ts[2,9]
   - Conv_1_Relu_ss[2,9] = 0
 Conv_1_Relu_constraint_1[2,10]: - Conv_1_t[2,10] + Conv_1_Relu_ts[2,10]
   - Conv_1_Relu_ss[2,10] = 0
 Conv_1_Relu_constraint_1[2,11]: - Conv_1_t[2,11] + Conv_1_Relu_ts[2,11]
   - Conv_1_Relu_ss[2,11] = 0
 Conv_1_Relu_constraint_1[2,12]: - Conv_1_t[2,12] + Conv_1_Relu_ts[2,12]
   - Conv_1_Relu_ss[2,12] = 0
 Conv_1_Relu_constraint_1[2,13]: - Conv_1_t[2,13] + Conv_1_Relu_ts[2,13]
   - Conv_1_Relu_ss[2,13] = 0
 Conv_1_Relu_constraint_1[2,14]: - Conv_1_t[2,14] + Conv_1_Relu_ts[2,14]
   - Conv_1_Relu_ss[2,14] = 0
 Conv_1_Relu_constraint_1[2,15]: - Conv_1_t[2,15] + Conv_1_Relu_ts[2,15]
   - Conv_1_Relu_ss[2,15] = 0
 Conv_1_Relu_constraint_1[3,0]: - Conv_1_t[3,0] + Conv_1_Relu_ts[3,0]
   - Conv_1_Relu_ss[3,0] = 0
 Conv_1_Relu_constraint_1[3,1]: - Conv_1_t[3,1] + Conv_1_Relu_ts[3,1]
   - Conv_1_Relu_ss[3,1] = 0
 Conv_1_Relu_constraint_1[3,2]: - Conv_1_t[3,2] + Conv_1_Relu_ts[3,2]
   - Conv_1_Relu_ss[3,2] = 0
 Conv_1_Relu_constraint_1[3,3]: - Conv_1_t[3,3] + Conv_1_Relu_ts[3,3]
   - Conv_1_Relu_ss[3,3] = 0
 Conv_1_Relu_constraint_1[3,4]: - Conv_1_t[3,4] + Conv_1_Relu_ts[3,4]
   - Conv_1_Relu_ss[3,4] = 0
 Conv_1_Relu_constraint_1[3,5]: - Conv_1_t[3,5] + Conv_1_Relu_ts[3,5]
   - Conv_1_Relu_ss[3,5] = 0
 Conv_1_Relu_constraint_1[3,6]: - Conv_1_t[3,6] + Conv_1_Relu_ts[3,6]
   - Conv_1_Relu_ss[3,6] = 0
 Conv_1_Relu_constraint_1[3,7]: - Conv_1_t[3,7] + Conv_1_Relu_ts[3,7]
   - Conv_1_Relu_ss[3,7] = 0
 Conv_1_Relu_constraint_1[3,8]: - Conv_1_t[3,8] + Conv_1_Relu_ts[3,8]
   - Conv_1_Relu_ss[3,8] = 0
 Conv_1_Relu_constraint_1[3,9]: - Conv_1_t[3,9] + Conv_1_Relu_ts[3,9]
   - Conv_1_Relu_ss[3,9] = 0
 Conv_1_Relu_constraint_1[3,10]: - Conv_1_t[3,10] + Conv_1_Relu_ts[3,10]
   - Conv_1_Relu_ss[3,10] = 0
 Conv_1_Relu_constraint_1[3,11]: - Conv_1_t[3,11] + Conv_1_Relu_ts[3,11]
   - Conv_1_Relu_ss[3,11] = 0
 Conv_1_Relu_constraint_1[3,12]: - Conv_1_t[3,12] + Conv_1_Relu_ts[3,12]
   - Conv_1_Relu_ss[3,12] = 0
 Conv_1_Relu_constraint_1[3,13]: - Conv_1_t[3,13] + Conv_1_Relu_ts[3,13]
   - Conv_1_Relu_ss[3,13] = 0
 Conv_1_Relu_constraint_1[3,14]: - Conv_1_t[3,14] + Conv_1_Relu_ts[3,14]
   - Conv_1_Relu_ss[3,14] = 0
 Conv_1_Relu_constraint_1[3,15]: - Conv_1_t[3,15] + Conv_1_Relu_ts[3,15]
   - Conv_1_Relu_ss[3,15] = 0
 Conv_1_Relu_constraint_1[4,0]: - Conv_1_t[4,0] + Conv_1_Relu_ts[4,0]
   - Conv_1_Relu_ss[4,0] = 0
 Conv_1_Relu_constraint_1[4,1]: - Conv_1_t[4,1] + Conv_1_Relu_ts[4,1]
   - Conv_1_Relu_ss[4,1] = 0
 Conv_1_Relu_constraint_1[4,2]: - Conv_1_t[4,2] + Conv_1_Relu_ts[4,2]
   - Conv_1_Relu_ss[4,2] = 0
 Conv_1_Relu_constraint_1[4,3]: - Conv_1_t[4,3] + Conv_1_Relu_ts[4,3]
   - Conv_1_Relu_ss[4,3] = 0
 Conv_1_Relu_constraint_1[4,4]: - Conv_1_t[4,4] + Conv_1_Relu_ts[4,4]
   - Conv_1_Relu_ss[4,4] = 0
 Conv_1_Relu_constraint_1[4,5]: - Conv_1_t[4,5] + Conv_1_Relu_ts[4,5]
   - Conv_1_Relu_ss[4,5] = 0
 Conv_1_Relu_constraint_1[4,6]: - Conv_1_t[4,6] + Conv_1_Relu_ts[4,6]
   - Conv_1_Relu_ss[4,6] = 0
 Conv_1_Relu_constraint_1[4,7]: - Conv_1_t[4,7] + Conv_1_Relu_ts[4,7]
   - Conv_1_Relu_ss[4,7] = 0
 Conv_1_Relu_constraint_1[4,8]: - Conv_1_t[4,8] + Conv_1_Relu_ts[4,8]
   - Conv_1_Relu_ss[4,8] = 0
 Conv_1_Relu_constraint_1[4,9]: - Conv_1_t[4,9] + Conv_1_Relu_ts[4,9]
   - Conv_1_Relu_ss[4,9] = 0
 Conv_1_Relu_constraint_1[4,10]: - Conv_1_t[4,10] + Conv_1_Relu_ts[4,10]
   - Conv_1_Relu_ss[4,10] = 0
 Conv_1_Relu_constraint_1[4,11]: - Conv_1_t[4,11] + Conv_1_Relu_ts[4,11]
   - Conv_1_Relu_ss[4,11] = 0
 Conv_1_Relu_constraint_1[4,12]: - Conv_1_t[4,12] + Conv_1_Relu_ts[4,12]
   - Conv_1_Relu_ss[4,12] = 0
 Conv_1_Relu_constraint_1[4,13]: - Conv_1_t[4,13] + Conv_1_Relu_ts[4,13]
   - Conv_1_Relu_ss[4,13] = 0
 Conv_1_Relu_constraint_1[4,14]: - Conv_1_t[4,14] + Conv_1_Relu_ts[4,14]
   - Conv_1_Relu_ss[4,14] = 0
 Conv_1_Relu_constraint_1[4,15]: - Conv_1_t[4,15] + Conv_1_Relu_ts[4,15]
   - Conv_1_Relu_ss[4,15] = 0
 Conv_1_Relu_constraint_1[5,0]: - Conv_1_t[5,0] + Conv_1_Relu_ts[5,0]
   - Conv_1_Relu_ss[5,0] = 0
 Conv_1_Relu_constraint_1[5,1]: - Conv_1_t[5,1] + Conv_1_Relu_ts[5,1]
   - Conv_1_Relu_ss[5,1] = 0
 Conv_1_Relu_constraint_1[5,2]: - Conv_1_t[5,2] + Conv_1_Relu_ts[5,2]
   - Conv_1_Relu_ss[5,2] = 0
 Conv_1_Relu_constraint_1[5,3]: - Conv_1_t[5,3] + Conv_1_Relu_ts[5,3]
   - Conv_1_Relu_ss[5,3] = 0
 Conv_1_Relu_constraint_1[5,4]: - Conv_1_t[5,4] + Conv_1_Relu_ts[5,4]
   - Conv_1_Relu_ss[5,4] = 0
 Conv_1_Relu_constraint_1[5,5]: - Conv_1_t[5,5] + Conv_1_Relu_ts[5,5]
   - Conv_1_Relu_ss[5,5] = 0
 Conv_1_Relu_constraint_1[5,6]: - Conv_1_t[5,6] + Conv_1_Relu_ts[5,6]
   - Conv_1_Relu_ss[5,6] = 0
 Conv_1_Relu_constraint_1[5,7]: - Conv_1_t[5,7] + Conv_1_Relu_ts[5,7]
   - Conv_1_Relu_ss[5,7] = 0
 Conv_1_Relu_constraint_1[5,8]: - Conv_1_t[5,8] + Conv_1_Relu_ts[5,8]
   - Conv_1_Relu_ss[5,8] = 0
 Conv_1_Relu_constraint_1[5,9]: - Conv_1_t[5,9] + Conv_1_Relu_ts[5,9]
   - Conv_1_Relu_ss[5,9] = 0
 Conv_1_Relu_constraint_1[5,10]: - Conv_1_t[5,10] + Conv_1_Relu_ts[5,10]
   - Conv_1_Relu_ss[5,10] = 0
 Conv_1_Relu_constraint_1[5,11]: - Conv_1_t[5,11] + Conv_1_Relu_ts[5,11]
   - Conv_1_Relu_ss[5,11] = 0
 Conv_1_Relu_constraint_1[5,12]: - Conv_1_t[5,12] + Conv_1_Relu_ts[5,12]
   - Conv_1_Relu_ss[5,12] = 0
 Conv_1_Relu_constraint_1[5,13]: - Conv_1_t[5,13] + Conv_1_Relu_ts[5,13]
   - Conv_1_Relu_ss[5,13] = 0
 Conv_1_Relu_constraint_1[5,14]: - Conv_1_t[5,14] + Conv_1_Relu_ts[5,14]
   - Conv_1_Relu_ss[5,14] = 0
 Conv_1_Relu_constraint_1[5,15]: - Conv_1_t[5,15] + Conv_1_Relu_ts[5,15]
   - Conv_1_Relu_ss[5,15] = 0
 Conv_1_Relu_constraint_1[6,0]: - Conv_1_t[6,0] + Conv_1_Relu_ts[6,0]
   - Conv_1_Relu_ss[6,0] = 0
 Conv_1_Relu_constraint_1[6,1]: - Conv_1_t[6,1] + Conv_1_Relu_ts[6,1]
   - Conv_1_Relu_ss[6,1] = 0
 Conv_1_Relu_constraint_1[6,2]: - Conv_1_t[6,2] + Conv_1_Relu_ts[6,2]
   - Conv_1_Relu_ss[6,2] = 0
 Conv_1_Relu_constraint_1[6,3]: - Conv_1_t[6,3] + Conv_1_Relu_ts[6,3]
   - Conv_1_Relu_ss[6,3] = 0
 Conv_1_Relu_constraint_1[6,4]: - Conv_1_t[6,4] + Conv_1_Relu_ts[6,4]
   - Conv_1_Relu_ss[6,4] = 0
 Conv_1_Relu_constraint_1[6,5]: - Conv_1_t[6,5] + Conv_1_Relu_ts[6,5]
   - Conv_1_Relu_ss[6,5] = 0
 Conv_1_Relu_constraint_1[6,6]: - Conv_1_t[6,6] + Conv_1_Relu_ts[6,6]
   - Conv_1_Relu_ss[6,6] = 0
 Conv_1_Relu_constraint_1[6,7]: - Conv_1_t[6,7] + Conv_1_Relu_ts[6,7]
   - Conv_1_Relu_ss[6,7] = 0
 Conv_1_Relu_constraint_1[6,8]: - Conv_1_t[6,8] + Conv_1_Relu_ts[6,8]
   - Conv_1_Relu_ss[6,8] = 0
 Conv_1_Relu_constraint_1[6,9]: - Conv_1_t[6,9] + Conv_1_Relu_ts[6,9]
   - Conv_1_Relu_ss[6,9] = 0
 Conv_1_Relu_constraint_1[6,10]: - Conv_1_t[6,10] + Conv_1_Relu_ts[6,10]
   - Conv_1_Relu_ss[6,10] = 0
 Conv_1_Relu_constraint_1[6,11]: - Conv_1_t[6,11] + Conv_1_Relu_ts[6,11]
   - Conv_1_Relu_ss[6,11] = 0
 Conv_1_Relu_constraint_1[6,12]: - Conv_1_t[6,12] + Conv_1_Relu_ts[6,12]
   - Conv_1_Relu_ss[6,12] = 0
 Conv_1_Relu_constraint_1[6,13]: - Conv_1_t[6,13] + Conv_1_Relu_ts[6,13]
   - Conv_1_Relu_ss[6,13] = 0
 Conv_1_Relu_constraint_1[6,14]: - Conv_1_t[6,14] + Conv_1_Relu_ts[6,14]
   - Conv_1_Relu_ss[6,14] = 0
 Conv_1_Relu_constraint_1[6,15]: - Conv_1_t[6,15] + Conv_1_Relu_ts[6,15]
   - Conv_1_Relu_ss[6,15] = 0
 Conv_1_Relu_constraint_1[7,0]: - Conv_1_t[7,0] + Conv_1_Relu_ts[7,0]
   - Conv_1_Relu_ss[7,0] = 0
 Conv_1_Relu_constraint_1[7,1]: - Conv_1_t[7,1] + Conv_1_Relu_ts[7,1]
   - Conv_1_Relu_ss[7,1] = 0
 Conv_1_Relu_constraint_1[7,2]: - Conv_1_t[7,2] + Conv_1_Relu_ts[7,2]
   - Conv_1_Relu_ss[7,2] = 0
 Conv_1_Relu_constraint_1[7,3]: - Conv_1_t[7,3] + Conv_1_Relu_ts[7,3]
   - Conv_1_Relu_ss[7,3] = 0
 Conv_1_Relu_constraint_1[7,4]: - Conv_1_t[7,4] + Conv_1_Relu_ts[7,4]
   - Conv_1_Relu_ss[7,4] = 0
 Conv_1_Relu_constraint_1[7,5]: - Conv_1_t[7,5] + Conv_1_Relu_ts[7,5]
   - Conv_1_Relu_ss[7,5] = 0
 Conv_1_Relu_constraint_1[7,6]: - Conv_1_t[7,6] + Conv_1_Relu_ts[7,6]
   - Conv_1_Relu_ss[7,6] = 0
 Conv_1_Relu_constraint_1[7,7]: - Conv_1_t[7,7] + Conv_1_Relu_ts[7,7]
   - Conv_1_Relu_ss[7,7] = 0
 Conv_1_Relu_constraint_1[7,8]: - Conv_1_t[7,8] + Conv_1_Relu_ts[7,8]
   - Conv_1_Relu_ss[7,8] = 0
 Conv_1_Relu_constraint_1[7,9]: - Conv_1_t[7,9] + Conv_1_Relu_ts[7,9]
   - Conv_1_Relu_ss[7,9] = 0
 Conv_1_Relu_constraint_1[7,10]: - Conv_1_t[7,10] + Conv_1_Relu_ts[7,10]
   - Conv_1_Relu_ss[7,10] = 0
 Conv_1_Relu_constraint_1[7,11]: - Conv_1_t[7,11] + Conv_1_Relu_ts[7,11]
   - Conv_1_Relu_ss[7,11] = 0
 Conv_1_Relu_constraint_1[7,12]: - Conv_1_t[7,12] + Conv_1_Relu_ts[7,12]
   - Conv_1_Relu_ss[7,12] = 0
 Conv_1_Relu_constraint_1[7,13]: - Conv_1_t[7,13] + Conv_1_Relu_ts[7,13]
   - Conv_1_Relu_ss[7,13] = 0
 Conv_1_Relu_constraint_1[7,14]: - Conv_1_t[7,14] + Conv_1_Relu_ts[7,14]
   - Conv_1_Relu_ss[7,14] = 0
 Conv_1_Relu_constraint_1[7,15]: - Conv_1_t[7,15] + Conv_1_Relu_ts[7,15]
   - Conv_1_Relu_ss[7,15] = 0
 Conv_1_Relu_constraint_2[0,0]: Conv_1_Relu_ts[0,0] <= 0
 Conv_1_Relu_constraint_2[0,1]: Conv_1_Relu_ts[0,1]
   - 14.59003230898199 C1105 <= 0
 Conv_1_Relu_constraint_2[0,2]: Conv_1_Relu_ts[0,2] <= 0
 Conv_1_Relu_constraint_2[0,3]: Conv_1_Relu_ts[0,3] <= 0
 Conv_1_Relu_constraint_2[0,4]: Conv_1_Relu_ts[0,4] <= 0
 Conv_1_Relu_constraint_2[0,5]: Conv_1_Relu_ts[0,5] <= 0
 Conv_1_Relu_constraint_2[0,6]: Conv_1_Relu_ts[0,6] <= 0
 Conv_1_Relu_constraint_2[0,7]: Conv_1_Relu_ts[0,7] <= 0
 Conv_1_Relu_constraint_2[0,8]: Conv_1_Relu_ts[0,8] - 2.75325e-13 C1112
   <= 0
 Conv_1_Relu_constraint_2[0,9]: Conv_1_Relu_ts[0,9]
   - 8.234055808249286 C1113 <= 0
 Conv_1_Relu_constraint_2[0,10]: Conv_1_Relu_ts[0,10] <= 0
 Conv_1_Relu_constraint_2[0,11]: Conv_1_Relu_ts[0,11] - 3.07161e-05 C1115
   <= 0
 Conv_1_Relu_constraint_2[0,12]: Conv_1_Relu_ts[0,12] <= 0
 Conv_1_Relu_constraint_2[0,13]: Conv_1_Relu_ts[0,13]
   - 11.92005955570159 C1117 <= 0
 Conv_1_Relu_constraint_2[0,14]: Conv_1_Relu_ts[0,14]
   - 0.00295504399667217 C1118 <= 0
 Conv_1_Relu_constraint_2[0,15]: Conv_1_Relu_ts[0,15] <= 0
 Conv_1_Relu_constraint_2[1,0]: Conv_1_Relu_ts[1,0] <= 0
 Conv_1_Relu_constraint_2[1,1]: Conv_1_Relu_ts[1,1]
   - 14.59003230898199 C1121 <= 0
 Conv_1_Relu_constraint_2[1,2]: Conv_1_Relu_ts[1,2] <= 0
 Conv_1_Relu_constraint_2[1,3]: Conv_1_Relu_ts[1,3] <= 0
 Conv_1_Relu_constraint_2[1,4]: Conv_1_Relu_ts[1,4] <= 0
 Conv_1_Relu_constraint_2[1,5]: Conv_1_Relu_ts[1,5] <= 0
 Conv_1_Relu_constraint_2[1,6]: Conv_1_Relu_ts[1,6] <= 0
 Conv_1_Relu_constraint_2[1,7]: Conv_1_Relu_ts[1,7] <= 0
 Conv_1_Relu_constraint_2[1,8]: Conv_1_Relu_ts[1,8] - 2.75325e-13 C1128
   <= 0
 Conv_1_Relu_constraint_2[1,9]: Conv_1_Relu_ts[1,9]
   - 8.234055808249286 C1129 <= 0
 Conv_1_Relu_constraint_2[1,10]: Conv_1_Relu_ts[1,10] <= 0
 Conv_1_Relu_constraint_2[1,11]: Conv_1_Relu_ts[1,11] - 3.07161e-05 C1131
   <= 0
 Conv_1_Relu_constraint_2[1,12]: Conv_1_Relu_ts[1,12] <= 0
 Conv_1_Relu_constraint_2[1,13]: Conv_1_Relu_ts[1,13]
   - 11.92005955570159 C1133 <= 0
 Conv_1_Relu_constraint_2[1,14]: Conv_1_Relu_ts[1,14]
   - 0.00295504399667217 C1134 <= 0
 Conv_1_Relu_constraint_2[1,15]: Conv_1_Relu_ts[1,15] <= 0
 Conv_1_Relu_constraint_2[2,0]: Conv_1_Relu_ts[2,0] <= 0
 Conv_1_Relu_constraint_2[2,1]: Conv_1_Relu_ts[2,1]
   - 14.59003230898199 C1137 <= 0
 Conv_1_Relu_constraint_2[2,2]: Conv_1_Relu_ts[2,2] <= 0
 Conv_1_Relu_constraint_2[2,3]: Conv_1_Relu_ts[2,3] <= 0
 Conv_1_Relu_constraint_2[2,4]: Conv_1_Relu_ts[2,4] <= 0
 Conv_1_Relu_constraint_2[2,5]: Conv_1_Relu_ts[2,5] <= 0
 Conv_1_Relu_constraint_2[2,6]: Conv_1_Relu_ts[2,6] <= 0
 Conv_1_Relu_constraint_2[2,7]: Conv_1_Relu_ts[2,7] <= 0
 Conv_1_Relu_constraint_2[2,8]: Conv_1_Relu_ts[2,8] - 2.75325e-13 C1144
   <= 0
 Conv_1_Relu_constraint_2[2,9]: Conv_1_Relu_ts[2,9]
   - 8.234055808249286 C1145 <= 0
 Conv_1_Relu_constraint_2[2,10]: Conv_1_Relu_ts[2,10] <= 0
 Conv_1_Relu_constraint_2[2,11]: Conv_1_Relu_ts[2,11] - 3.07161e-05 C1147
   <= 0
 Conv_1_Relu_constraint_2[2,12]: Conv_1_Relu_ts[2,12] <= 0
 Conv_1_Relu_constraint_2[2,13]: Conv_1_Relu_ts[2,13]
   - 11.92005955570159 C1149 <= 0
 Conv_1_Relu_constraint_2[2,14]: Conv_1_Relu_ts[2,14]
   - 0.00295504399667217 C1150 <= 0
 Conv_1_Relu_constraint_2[2,15]: Conv_1_Relu_ts[2,15] <= 0
 Conv_1_Relu_constraint_2[3,0]: Conv_1_Relu_ts[3,0] <= 0
 Conv_1_Relu_constraint_2[3,1]: Conv_1_Relu_ts[3,1]
   - 14.59003230898199 C1153 <= 0
 Conv_1_Relu_constraint_2[3,2]: Conv_1_Relu_ts[3,2] <= 0
 Conv_1_Relu_constraint_2[3,3]: Conv_1_Relu_ts[3,3] <= 0
 Conv_1_Relu_constraint_2[3,4]: Conv_1_Relu_ts[3,4] <= 0
 Conv_1_Relu_constraint_2[3,5]: Conv_1_Relu_ts[3,5] <= 0
 Conv_1_Relu_constraint_2[3,6]: Conv_1_Relu_ts[3,6] <= 0
 Conv_1_Relu_constraint_2[3,7]: Conv_1_Relu_ts[3,7] <= 0
 Conv_1_Relu_constraint_2[3,8]: Conv_1_Relu_ts[3,8] - 2.75325e-13 C1160
   <= 0
 Conv_1_Relu_constraint_2[3,9]: Conv_1_Relu_ts[3,9]
   - 8.234055808249286 C1161 <= 0
 Conv_1_Relu_constraint_2[3,10]: Conv_1_Relu_ts[3,10] <= 0
 Conv_1_Relu_constraint_2[3,11]: Conv_1_Relu_ts[3,11] - 3.07161e-05 C1163
   <= 0
 Conv_1_Relu_constraint_2[3,12]: Conv_1_Relu_ts[3,12] <= 0
 Conv_1_Relu_constraint_2[3,13]: Conv_1_Relu_ts[3,13]
   - 11.92005955570159 C1165 <= 0
 Conv_1_Relu_constraint_2[3,14]: Conv_1_Relu_ts[3,14]
   - 0.00295504399667217 C1166 <= 0
 Conv_1_Relu_constraint_2[3,15]: Conv_1_Relu_ts[3,15] <= 0
 Conv_1_Relu_constraint_2[4,0]: Conv_1_Relu_ts[4,0] <= 0
 Conv_1_Relu_constraint_2[4,1]: Conv_1_Relu_ts[4,1]
   - 14.59003230898199 C1169 <= 0
 Conv_1_Relu_constraint_2[4,2]: Conv_1_Relu_ts[4,2] <= 0
 Conv_1_Relu_constraint_2[4,3]: Conv_1_Relu_ts[4,3] <= 0
 Conv_1_Relu_constraint_2[4,4]: Conv_1_Relu_ts[4,4] <= 0
 Conv_1_Relu_constraint_2[4,5]: Conv_1_Relu_ts[4,5] <= 0
 Conv_1_Relu_constraint_2[4,6]: Conv_1_Relu_ts[4,6] <= 0
 Conv_1_Relu_constraint_2[4,7]: Conv_1_Relu_ts[4,7] <= 0
 Conv_1_Relu_constraint_2[4,8]: Conv_1_Relu_ts[4,8] - 2.75325e-13 C1176
   <= 0
 Conv_1_Relu_constraint_2[4,9]: Conv_1_Relu_ts[4,9]
   - 8.234055808249286 C1177 <= 0
 Conv_1_Relu_constraint_2[4,10]: Conv_1_Relu_ts[4,10] <= 0
 Conv_1_Relu_constraint_2[4,11]: Conv_1_Relu_ts[4,11] - 3.07161e-05 C1179
   <= 0
 Conv_1_Relu_constraint_2[4,12]: Conv_1_Relu_ts[4,12] <= 0
 Conv_1_Relu_constraint_2[4,13]: Conv_1_Relu_ts[4,13]
   - 11.92005955570159 C1181 <= 0
 Conv_1_Relu_constraint_2[4,14]: Conv_1_Relu_ts[4,14]
   - 0.00295504399667217 C1182 <= 0
 Conv_1_Relu_constraint_2[4,15]: Conv_1_Relu_ts[4,15] <= 0
 Conv_1_Relu_constraint_2[5,0]: Conv_1_Relu_ts[5,0] <= 0
 Conv_1_Relu_constraint_2[5,1]: Conv_1_Relu_ts[5,1]
   - 14.59003230898199 C1185 <= 0
 Conv_1_Relu_constraint_2[5,2]: Conv_1_Relu_ts[5,2] <= 0
 Conv_1_Relu_constraint_2[5,3]: Conv_1_Relu_ts[5,3] <= 0
 Conv_1_Relu_constraint_2[5,4]: Conv_1_Relu_ts[5,4] <= 0
 Conv_1_Relu_constraint_2[5,5]: Conv_1_Relu_ts[5,5] <= 0
 Conv_1_Relu_constraint_2[5,6]: Conv_1_Relu_ts[5,6] <= 0
 Conv_1_Relu_constraint_2[5,7]: Conv_1_Relu_ts[5,7] <= 0
 Conv_1_Relu_constraint_2[5,8]: Conv_1_Relu_ts[5,8] - 2.75325e-13 C1192
   <= 0
 Conv_1_Relu_constraint_2[5,9]: Conv_1_Relu_ts[5,9]
   - 8.234055808249286 C1193 <= 0
 Conv_1_Relu_constraint_2[5,10]: Conv_1_Relu_ts[5,10] <= 0
 Conv_1_Relu_constraint_2[5,11]: Conv_1_Relu_ts[5,11] - 3.07161e-05 C1195
   <= 0
 Conv_1_Relu_constraint_2[5,12]: Conv_1_Relu_ts[5,12] <= 0
 Conv_1_Relu_constraint_2[5,13]: Conv_1_Relu_ts[5,13]
   - 11.92005955570159 C1197 <= 0
 Conv_1_Relu_constraint_2[5,14]: Conv_1_Relu_ts[5,14]
   - 0.00295504399667217 C1198 <= 0
 Conv_1_Relu_constraint_2[5,15]: Conv_1_Relu_ts[5,15] <= 0
 Conv_1_Relu_constraint_2[6,0]: Conv_1_Relu_ts[6,0] <= 0
 Conv_1_Relu_constraint_2[6,1]: Conv_1_Relu_ts[6,1]
   - 14.59003230898199 C1201 <= 0
 Conv_1_Relu_constraint_2[6,2]: Conv_1_Relu_ts[6,2] <= 0
 Conv_1_Relu_constraint_2[6,3]: Conv_1_Relu_ts[6,3] <= 0
 Conv_1_Relu_constraint_2[6,4]: Conv_1_Relu_ts[6,4] <= 0
 Conv_1_Relu_constraint_2[6,5]: Conv_1_Relu_ts[6,5] <= 0
 Conv_1_Relu_constraint_2[6,6]: Conv_1_Relu_ts[6,6] <= 0
 Conv_1_Relu_constraint_2[6,7]: Conv_1_Relu_ts[6,7] <= 0
 Conv_1_Relu_constraint_2[6,8]: Conv_1_Relu_ts[6,8] - 2.75325e-13 C1208
   <= 0
 Conv_1_Relu_constraint_2[6,9]: Conv_1_Relu_ts[6,9]
   - 8.234055808249286 C1209 <= 0
 Conv_1_Relu_constraint_2[6,10]: Conv_1_Relu_ts[6,10] <= 0
 Conv_1_Relu_constraint_2[6,11]: Conv_1_Relu_ts[6,11] - 3.07161e-05 C1211
   <= 0
 Conv_1_Relu_constraint_2[6,12]: Conv_1_Relu_ts[6,12] <= 0
 Conv_1_Relu_constraint_2[6,13]: Conv_1_Relu_ts[6,13]
   - 11.92005955570159 C1213 <= 0
 Conv_1_Relu_constraint_2[6,14]: Conv_1_Relu_ts[6,14]
   - 0.00295504399667217 C1214 <= 0
 Conv_1_Relu_constraint_2[6,15]: Conv_1_Relu_ts[6,15] <= 0
 Conv_1_Relu_constraint_2[7,0]: Conv_1_Relu_ts[7,0] <= 0
 Conv_1_Relu_constraint_2[7,1]: Conv_1_Relu_ts[7,1]
   - 14.59003230898199 C1217 <= 0
 Conv_1_Relu_constraint_2[7,2]: Conv_1_Relu_ts[7,2] <= 0
 Conv_1_Relu_constraint_2[7,3]: Conv_1_Relu_ts[7,3] <= 0
 Conv_1_Relu_constraint_2[7,4]: Conv_1_Relu_ts[7,4] <= 0
 Conv_1_Relu_constraint_2[7,5]: Conv_1_Relu_ts[7,5] <= 0
 Conv_1_Relu_constraint_2[7,6]: Conv_1_Relu_ts[7,6] <= 0
 Conv_1_Relu_constraint_2[7,7]: Conv_1_Relu_ts[7,7] <= 0
 Conv_1_Relu_constraint_2[7,8]: Conv_1_Relu_ts[7,8] - 2.75325e-13 C1224
   <= 0
 Conv_1_Relu_constraint_2[7,9]: Conv_1_Relu_ts[7,9]
   - 8.234055808249286 C1225 <= 0
 Conv_1_Relu_constraint_2[7,10]: Conv_1_Relu_ts[7,10] <= 0
 Conv_1_Relu_constraint_2[7,11]: Conv_1_Relu_ts[7,11] - 3.07161e-05 C1227
   <= 0
 Conv_1_Relu_constraint_2[7,12]: Conv_1_Relu_ts[7,12] <= 0
 Conv_1_Relu_constraint_2[7,13]: Conv_1_Relu_ts[7,13]
   - 11.92005955570159 C1229 <= 0
 Conv_1_Relu_constraint_2[7,14]: Conv_1_Relu_ts[7,14]
   - 0.00295504399667217 C1230 <= 0
 Conv_1_Relu_constraint_2[7,15]: Conv_1_Relu_ts[7,15] <= 0
 Conv_1_Relu_constraint_3[0,0]: Conv_1_Relu_ss[0,0]
   + 0.00554961914110762 C1104 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[0,1]: Conv_1_Relu_ss[0,1]
   + 1.857019803605887 C1105 <= 1.857019803605887
 Conv_1_Relu_constraint_3[0,2]: Conv_1_Relu_ss[0,2] + 1.05164e-07 C1106
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[0,3]: Conv_1_Relu_ss[0,3]
   + 0.00336194144980918 C1107 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[0,4]: Conv_1_Relu_ss[0,4] + 3.66243e-06 C1108
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[0,5]: Conv_1_Relu_ss[0,5] + 2.93491e-08 C1109
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[0,6]: Conv_1_Relu_ss[0,6]
   + 5.8459259755839165e-04 C1110 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[0,7]: Conv_1_Relu_ss[0,7] + 1.08295e-07 C1111
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[0,8]: Conv_1_Relu_ss[0,8] + 1.27821e-12 C1112
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[0,9]: Conv_1_Relu_ss[0,9]
   + 4.958788286540738 C1113 <= 4.958788286540738
 Conv_1_Relu_constraint_3[0,10]: Conv_1_Relu_ss[0,10] + 1.45772e-05 C1114
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[0,11]: Conv_1_Relu_ss[0,11]
   + 0.00236660145696522 C1115 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[0,12]: Conv_1_Relu_ss[0,12] + 3.08689e-07 C1116
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[0,13]: Conv_1_Relu_ss[0,13]
   + 4.431268808214668 C1117 <= 4.431268808214668
 Conv_1_Relu_constraint_3[0,14]: Conv_1_Relu_ss[0,14]
   + 0.178791987482969 C1118 <= 0.178791987482969
 Conv_1_Relu_constraint_3[0,15]: Conv_1_Relu_ss[0,15] + 1.22225e-07 C1119
   <= 1.22225e-07
 Conv_1_Relu_constraint_3[1,0]: Conv_1_Relu_ss[1,0]
   + 0.00554961914110762 C1120 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[1,1]: Conv_1_Relu_ss[1,1]
   + 1.857019803605887 C1121 <= 1.857019803605887
 Conv_1_Relu_constraint_3[1,2]: Conv_1_Relu_ss[1,2] + 1.05164e-07 C1122
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[1,3]: Conv_1_Relu_ss[1,3]
   + 0.00336194144980918 C1123 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[1,4]: Conv_1_Relu_ss[1,4] + 3.66243e-06 C1124
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[1,5]: Conv_1_Relu_ss[1,5] + 2.93491e-08 C1125
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[1,6]: Conv_1_Relu_ss[1,6]
   + 5.8459259755839165e-04 C1126 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[1,7]: Conv_1_Relu_ss[1,7] + 1.08295e-07 C1127
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[1,8]: Conv_1_Relu_ss[1,8] + 1.27821e-12 C1128
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[1,9]: Conv_1_Relu_ss[1,9]
   + 4.958788286540738 C1129 <= 4.958788286540738
 Conv_1_Relu_constraint_3[1,10]: Conv_1_Relu_ss[1,10] + 1.45772e-05 C1130
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[1,11]: Conv_1_Relu_ss[1,11]
   + 0.00236660145696522 C1131 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[1,12]: Conv_1_Relu_ss[1,12] + 3.08689e-07 C1132
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[1,13]: Conv_1_Relu_ss[1,13]
   + 4.431268808214668 C1133 <= 4.431268808214668
 Conv_1_Relu_constraint_3[1,14]: Conv_1_Relu_ss[1,14]
   + 0.178791987482969 C1134 <= 0.178791987482969
 Conv_1_Relu_constraint_3[1,15]: Conv_1_Relu_ss[1,15] + 1.22225e-07 C1135
   <= 1.22225e-07
 Conv_1_Relu_constraint_3[2,0]: Conv_1_Relu_ss[2,0]
   + 0.00554961914110762 C1136 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[2,1]: Conv_1_Relu_ss[2,1]
   + 1.857019803605887 C1137 <= 1.857019803605887
 Conv_1_Relu_constraint_3[2,2]: Conv_1_Relu_ss[2,2] + 1.05164e-07 C1138
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[2,3]: Conv_1_Relu_ss[2,3]
   + 0.00336194144980918 C1139 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[2,4]: Conv_1_Relu_ss[2,4] + 3.66243e-06 C1140
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[2,5]: Conv_1_Relu_ss[2,5] + 2.93491e-08 C1141
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[2,6]: Conv_1_Relu_ss[2,6]
   + 5.8459259755839165e-04 C1142 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[2,7]: Conv_1_Relu_ss[2,7] + 1.08295e-07 C1143
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[2,8]: Conv_1_Relu_ss[2,8] + 1.27821e-12 C1144
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[2,9]: Conv_1_Relu_ss[2,9]
   + 4.958788286540738 C1145 <= 4.958788286540738
 Conv_1_Relu_constraint_3[2,10]: Conv_1_Relu_ss[2,10] + 1.45772e-05 C1146
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[2,11]: Conv_1_Relu_ss[2,11]
   + 0.00236660145696522 C1147 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[2,12]: Conv_1_Relu_ss[2,12] + 3.08689e-07 C1148
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[2,13]: Conv_1_Relu_ss[2,13]
   + 4.431268808214668 C1149 <= 4.431268808214668
 Conv_1_Relu_constraint_3[2,14]: Conv_1_Relu_ss[2,14]
   + 0.178791987482969 C1150 <= 0.178791987482969
 Conv_1_Relu_constraint_3[2,15]: Conv_1_Relu_ss[2,15] + 1.22225e-07 C1151
   <= 1.22225e-07
 Conv_1_Relu_constraint_3[3,0]: Conv_1_Relu_ss[3,0]
   + 0.00554961914110762 C1152 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[3,1]: Conv_1_Relu_ss[3,1]
   + 1.857019803605887 C1153 <= 1.857019803605887
 Conv_1_Relu_constraint_3[3,2]: Conv_1_Relu_ss[3,2] + 1.05164e-07 C1154
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[3,3]: Conv_1_Relu_ss[3,3]
   + 0.00336194144980918 C1155 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[3,4]: Conv_1_Relu_ss[3,4] + 3.66243e-06 C1156
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[3,5]: Conv_1_Relu_ss[3,5] + 2.93491e-08 C1157
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[3,6]: Conv_1_Relu_ss[3,6]
   + 5.8459259755839165e-04 C1158 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[3,7]: Conv_1_Relu_ss[3,7] + 1.08295e-07 C1159
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[3,8]: Conv_1_Relu_ss[3,8] + 1.27821e-12 C1160
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[3,9]: Conv_1_Relu_ss[3,9]
   + 4.958788286540738 C1161 <= 4.958788286540738
 Conv_1_Relu_constraint_3[3,10]: Conv_1_Relu_ss[3,10] + 1.45772e-05 C1162
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[3,11]: Conv_1_Relu_ss[3,11]
   + 0.00236660145696522 C1163 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[3,12]: Conv_1_Relu_ss[3,12] + 3.08689e-07 C1164
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[3,13]: Conv_1_Relu_ss[3,13]
   + 4.431268808214668 C1165 <= 4.431268808214668
 Conv_1_Relu_constraint_3[3,14]: Conv_1_Relu_ss[3,14]
   + 0.178791987482969 C1166 <= 0.178791987482969
 Conv_1_Relu_constraint_3[3,15]: Conv_1_Relu_ss[3,15] + 1.22225e-07 C1167
   <= 1.22225e-07
 Conv_1_Relu_constraint_3[4,0]: Conv_1_Relu_ss[4,0]
   + 0.00554961914110762 C1168 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[4,1]: Conv_1_Relu_ss[4,1]
   + 1.857019803605887 C1169 <= 1.857019803605887
 Conv_1_Relu_constraint_3[4,2]: Conv_1_Relu_ss[4,2] + 1.05164e-07 C1170
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[4,3]: Conv_1_Relu_ss[4,3]
   + 0.00336194144980918 C1171 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[4,4]: Conv_1_Relu_ss[4,4] + 3.66243e-06 C1172
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[4,5]: Conv_1_Relu_ss[4,5] + 2.93491e-08 C1173
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[4,6]: Conv_1_Relu_ss[4,6]
   + 5.8459259755839165e-04 C1174 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[4,7]: Conv_1_Relu_ss[4,7] + 1.08295e-07 C1175
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[4,8]: Conv_1_Relu_ss[4,8] + 1.27821e-12 C1176
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[4,9]: Conv_1_Relu_ss[4,9]
   + 4.958788286540738 C1177 <= 4.958788286540738
 Conv_1_Relu_constraint_3[4,10]: Conv_1_Relu_ss[4,10] + 1.45772e-05 C1178
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[4,11]: Conv_1_Relu_ss[4,11]
   + 0.00236660145696522 C1179 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[4,12]: Conv_1_Relu_ss[4,12] + 3.08689e-07 C1180
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[4,13]: Conv_1_Relu_ss[4,13]
   + 4.431268808214668 C1181 <= 4.431268808214668
 Conv_1_Relu_constraint_3[4,14]: Conv_1_Relu_ss[4,14]
   + 0.178791987482969 C1182 <= 0.178791987482969
 Conv_1_Relu_constraint_3[4,15]: Conv_1_Relu_ss[4,15] + 1.22225e-07 C1183
   <= 1.22225e-07
 Conv_1_Relu_constraint_3[5,0]: Conv_1_Relu_ss[5,0]
   + 0.00554961914110762 C1184 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[5,1]: Conv_1_Relu_ss[5,1]
   + 1.857019803605887 C1185 <= 1.857019803605887
 Conv_1_Relu_constraint_3[5,2]: Conv_1_Relu_ss[5,2] + 1.05164e-07 C1186
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[5,3]: Conv_1_Relu_ss[5,3]
   + 0.00336194144980918 C1187 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[5,4]: Conv_1_Relu_ss[5,4] + 3.66243e-06 C1188
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[5,5]: Conv_1_Relu_ss[5,5] + 2.93491e-08 C1189
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[5,6]: Conv_1_Relu_ss[5,6]
   + 5.8459259755839165e-04 C1190 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[5,7]: Conv_1_Relu_ss[5,7] + 1.08295e-07 C1191
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[5,8]: Conv_1_Relu_ss[5,8] + 1.27821e-12 C1192
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[5,9]: Conv_1_Relu_ss[5,9]
   + 4.958788286540738 C1193 <= 4.958788286540738
 Conv_1_Relu_constraint_3[5,10]: Conv_1_Relu_ss[5,10] + 1.45772e-05 C1194
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[5,11]: Conv_1_Relu_ss[5,11]
   + 0.00236660145696522 C1195 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[5,12]: Conv_1_Relu_ss[5,12] + 3.08689e-07 C1196
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[5,13]: Conv_1_Relu_ss[5,13]
   + 4.431268808214668 C1197 <= 4.431268808214668
 Conv_1_Relu_constraint_3[5,14]: Conv_1_Relu_ss[5,14]
   + 0.178791987482969 C1198 <= 0.178791987482969
 Conv_1_Relu_constraint_3[5,15]: Conv_1_Relu_ss[5,15] + 1.22225e-07 C1199
   <= 1.22225e-07
 Conv_1_Relu_constraint_3[6,0]: Conv_1_Relu_ss[6,0]
   + 0.00554961914110762 C1200 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[6,1]: Conv_1_Relu_ss[6,1]
   + 1.857019803605887 C1201 <= 1.857019803605887
 Conv_1_Relu_constraint_3[6,2]: Conv_1_Relu_ss[6,2] + 1.05164e-07 C1202
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[6,3]: Conv_1_Relu_ss[6,3]
   + 0.00336194144980918 C1203 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[6,4]: Conv_1_Relu_ss[6,4] + 3.66243e-06 C1204
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[6,5]: Conv_1_Relu_ss[6,5] + 2.93491e-08 C1205
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[6,6]: Conv_1_Relu_ss[6,6]
   + 5.8459259755839165e-04 C1206 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[6,7]: Conv_1_Relu_ss[6,7] + 1.08295e-07 C1207
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[6,8]: Conv_1_Relu_ss[6,8] + 1.27821e-12 C1208
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[6,9]: Conv_1_Relu_ss[6,9]
   + 4.958788286540738 C1209 <= 4.958788286540738
 Conv_1_Relu_constraint_3[6,10]: Conv_1_Relu_ss[6,10] + 1.45772e-05 C1210
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[6,11]: Conv_1_Relu_ss[6,11]
   + 0.00236660145696522 C1211 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[6,12]: Conv_1_Relu_ss[6,12] + 3.08689e-07 C1212
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[6,13]: Conv_1_Relu_ss[6,13]
   + 4.431268808214668 C1213 <= 4.431268808214668
 Conv_1_Relu_constraint_3[6,14]: Conv_1_Relu_ss[6,14]
   + 0.178791987482969 C1214 <= 0.178791987482969
 Conv_1_Relu_constraint_3[6,15]: Conv_1_Relu_ss[6,15] + 1.22225e-07 C1215
   <= 1.22225e-07
 Conv_1_Relu_constraint_3[7,0]: Conv_1_Relu_ss[7,0]
   + 0.00554961914110762 C1216 <= 0.00554961914110762
 Conv_1_Relu_constraint_3[7,1]: Conv_1_Relu_ss[7,1]
   + 1.857019803605887 C1217 <= 1.857019803605887
 Conv_1_Relu_constraint_3[7,2]: Conv_1_Relu_ss[7,2] + 1.05164e-07 C1218
   <= 1.05164e-07
 Conv_1_Relu_constraint_3[7,3]: Conv_1_Relu_ss[7,3]
   + 0.00336194144980918 C1219 <= 0.00336194144980918
 Conv_1_Relu_constraint_3[7,4]: Conv_1_Relu_ss[7,4] + 3.66243e-06 C1220
   <= 3.66243e-06
 Conv_1_Relu_constraint_3[7,5]: Conv_1_Relu_ss[7,5] + 2.93491e-08 C1221
   <= 2.93491e-08
 Conv_1_Relu_constraint_3[7,6]: Conv_1_Relu_ss[7,6]
   + 5.8459259755839165e-04 C1222 <= 5.8459259755839165e-04
 Conv_1_Relu_constraint_3[7,7]: Conv_1_Relu_ss[7,7] + 1.08295e-07 C1223
   <= 1.08295e-07
 Conv_1_Relu_constraint_3[7,8]: Conv_1_Relu_ss[7,8] + 1.27821e-12 C1224
   <= 1.27821e-12
 Conv_1_Relu_constraint_3[7,9]: Conv_1_Relu_ss[7,9]
   + 4.958788286540738 C1225 <= 4.958788286540738
 Conv_1_Relu_constraint_3[7,10]: Conv_1_Relu_ss[7,10] + 1.45772e-05 C1226
   <= 1.45772e-05
 Conv_1_Relu_constraint_3[7,11]: Conv_1_Relu_ss[7,11]
   + 0.00236660145696522 C1227 <= 0.00236660145696522
 Conv_1_Relu_constraint_3[7,12]: Conv_1_Relu_ss[7,12] + 3.08689e-07 C1228
   <= 3.08689e-07
 Conv_1_Relu_constraint_3[7,13]: Conv_1_Relu_ss[7,13]
   + 4.431268808214668 C1229 <= 4.431268808214668
 Conv_1_Relu_constraint_3[7,14]: Conv_1_Relu_ss[7,14]
   + 0.178791987482969 C1230 <= 0.178791987482969
 Conv_1_Relu_constraint_3[7,15]: Conv_1_Relu_ss[7,15] + 1.22225e-07 C1231
   <= 1.22225e-07
 Conv_2_output_constraint[0,0]: 5.62815e-10 Conv_1_Relu_ts[0,7]
   + 1.92775e-11 Conv_1_Relu_ts[0,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[0,15]
   + 5.32965e-11 Conv_2_aggregated_features[0,7]
   - 3.50224e-08 Conv_2_aggregated_features[0,15] + Conv_2_t[0,0]
   = 2.0494e-14
 Conv_2_output_constraint[0,1]: - 3.1209e-05 Conv_1_Relu_ts[0,1]
   + 3.2097e-12 Conv_1_Relu_ts[0,2] + 6.75828e-11 Conv_1_Relu_ts[0,4]
   + 6.05596e-11 Conv_1_Relu_ts[0,5] + 9.57194e-07 Conv_1_Relu_ts[0,6]
   - 1.94554e-13 Conv_1_Relu_ts[0,7] + 9.60207e-05 Conv_1_Relu_ts[0,9]
   + 3.54229e-11 Conv_1_Relu_ts[0,10] + 2.0034e-09 Conv_1_Relu_ts[0,11]
   + 3.41256e-05 Conv_1_Relu_ts[0,13] + 2.24541e-06 Conv_1_Relu_ts[0,14]
   - 7.56678e-10 Conv_1_Relu_ts[0,15]
   - 3.63258e-06 Conv_2_aggregated_features[0,1]
   - 2.02372e-12 Conv_2_aggregated_features[0,2]
   - 6.7717e-13 Conv_2_aggregated_features[0,4]
   + 1.64055e-10 Conv_2_aggregated_features[0,5]
   + 8.63851e-05 Conv_2_aggregated_features[0,6]
   - 5.43338e-10 Conv_2_aggregated_features[0,7]
   + 8.87988e-05 Conv_2_aggregated_features[0,9]
   + 2.03334e-11 Conv_2_aggregated_features[0,10]
   + 2.09865e-09 Conv_2_aggregated_features[0,11]
   + 5.1289e-05 Conv_2_aggregated_features[0,13]
   - 1.99215e-06 Conv_2_aggregated_features[0,14]
   - 2.85245e-06 Conv_2_aggregated_features[0,15] + Conv_2_t[0,1]
   = -5.09804e-05
 Conv_2_output_constraint[0,2]: 4.6554e-12 Conv_1_Relu_ts[0,1]
   - 5.08004e-07 Conv_1_Relu_ts[0,4] + 8.2088e-12 Conv_1_Relu_ts[0,9]
   + 3.89405e-05 Conv_1_Relu_ts[0,10] + 3.83811e-12 Conv_1_Relu_ts[0,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[0,15]
   + 2.16136e-12 Conv_2_aggregated_features[0,1]
   - 4.18879e-13 Conv_2_aggregated_features[0,4]
   + 1.01588e-11 Conv_2_aggregated_features[0,9]
   + 4.38408e-12 Conv_2_aggregated_features[0,13]
   + 6.90517e-06 Conv_2_aggregated_features[0,15] + Conv_2_t[0,2]
   = -1.21231e-11
 Conv_2_output_constraint[0,3]: - 9.77857e-07 Conv_1_Relu_ts[0,1]
   - 1.62341e-06 Conv_1_Relu_ts[0,4] - 1.26004e-09 Conv_1_Relu_ts[0,6]
   + 2.40003e-07 Conv_1_Relu_ts[0,7] - 5.41958e-07 Conv_1_Relu_ts[0,9]
   + 1.16363e-09 Conv_1_Relu_ts[0,10] - 2.13281e-11 Conv_1_Relu_ts[0,11]
   + 4.12955e-09 Conv_1_Relu_ts[0,12] - 3.71195e-07 Conv_1_Relu_ts[0,13]
   + 2.2241e-11 Conv_1_Relu_ts[0,14] + 6.37491e-09 Conv_1_Relu_ts[0,15]
   - 1.2731e-06 Conv_2_aggregated_features[0,1]
   - 1.0258e-06 Conv_2_aggregated_features[0,4]
   + 4.06095e-10 Conv_2_aggregated_features[0,6]
   + 3.12011e-07 Conv_2_aggregated_features[0,7]
   - 3.09535e-07 Conv_2_aggregated_features[0,9]
   - 0.00177046051248908 Conv_2_aggregated_features[0,10]
   + 8.54383e-13 Conv_2_aggregated_features[0,11]
   - 3.03582e-06 Conv_2_aggregated_features[0,12]
   - 4.41436e-07 Conv_2_aggregated_features[0,13]
   + 7.0584e-12 Conv_2_aggregated_features[0,14]
   - 9.96128e-05 Conv_2_aggregated_features[0,15] + Conv_2_t[0,3]
   = 4.02827e-07
 Conv_2_output_constraint[0,4]: 5.03273e-07 Conv_1_Relu_ts[0,1]
   - 6.22027e-09 Conv_1_Relu_ts[0,5] - 1.47914e-11 Conv_1_Relu_ts[0,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[0,9]
   + 1.18025e-13 Conv_1_Relu_ts[0,10] + 9.23352e-05 Conv_1_Relu_ts[0,11]
   + 0.000518991 Conv_1_Relu_ts[0,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[0,14]
   - 1.05921e-06 Conv_2_aggregated_features[0,1]
   + 2.88866e-10 Conv_2_aggregated_features[0,5]
   - 2.58968e-11 Conv_2_aggregated_features[0,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[0,9]
   - 3.93262e-05 Conv_2_aggregated_features[0,11]
   + 0.000690713 Conv_2_aggregated_features[0,13]
   + 7.70641e-05 Conv_2_aggregated_features[0,14] + Conv_2_t[0,4]
   = -5.84568e-07
 Conv_2_output_constraint[0,5]: 4.37405e-10 Conv_1_Relu_ts[0,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[0,4]
   + 1.77244e-09 Conv_1_Relu_ts[0,5] + 6.99595e-10 Conv_1_Relu_ts[0,9]
   - 1.77001e-13 Conv_1_Relu_ts[0,11] + 2.86208e-12 Conv_1_Relu_ts[0,12]
   + 3.24683e-10 Conv_1_Relu_ts[0,13] - 6.56311e-06 Conv_1_Relu_ts[0,15]
   - 4.63786e-09 Conv_2_aggregated_features[0,0]
   - 4.27797e-10 Conv_2_aggregated_features[0,1]
   - 2.50277e-13 Conv_2_aggregated_features[0,4]
   - 2.35888e-05 Conv_2_aggregated_features[0,5]
   + 6.12624e-10 Conv_2_aggregated_features[0,9]
   + 3.98546e-10 Conv_2_aggregated_features[0,13]
   + 9.89818e-10 Conv_2_aggregated_features[0,15] + Conv_2_t[0,5]
   = 6.24238e-10
 Conv_2_output_constraint[0,6]: - 4.80371e-09 Conv_1_Relu_ts[0,0]
   + 0.2580605149269104 Conv_1_Relu_ts[0,1]
   + 6.84534e-09 Conv_1_Relu_ts[0,2] + 7.44651e-09 Conv_1_Relu_ts[0,3]
   - 2.26729e-08 Conv_1_Relu_ts[0,4] + 1.08868e-09 Conv_1_Relu_ts[0,5]
   + 0.000151346 Conv_1_Relu_ts[0,6] - 1.07095e-09 Conv_1_Relu_ts[0,7]
   - 0.4001754522323608 Conv_1_Relu_ts[0,9]
   - 6.82816e-07 Conv_1_Relu_ts[0,10] - 3.7591e-05 Conv_1_Relu_ts[0,11]
   - 1.62843e-08 Conv_1_Relu_ts[0,12]
   + 1.447632431983948 Conv_1_Relu_ts[0,13]
   - 1.73427e-05 Conv_1_Relu_ts[0,14] + 2.28396e-09 Conv_1_Relu_ts[0,15]
   - 5.33024e-08 Conv_2_aggregated_features[0,0]
   + 0.8234332203865051 Conv_2_aggregated_features[0,1]
   + 6.68388e-09 Conv_2_aggregated_features[0,2]
   - 4.97866e-07 Conv_2_aggregated_features[0,3]
   - 2.61287e-08 Conv_2_aggregated_features[0,4]
   + 5.03108e-11 Conv_2_aggregated_features[0,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[0,6]
   - 1.69808e-09 Conv_2_aggregated_features[0,7]
   - 1.210363864898682 Conv_2_aggregated_features[0,9]
   - 7.75092e-07 Conv_2_aggregated_features[0,10]
   - 6.16735e-05 Conv_2_aggregated_features[0,11]
   - 1.9235e-08 Conv_2_aggregated_features[0,12]
   + 0.4373943507671356 Conv_2_aggregated_features[0,13]
   - 0.00871682073920965 Conv_2_aggregated_features[0,14]
   + 1.60146e-09 Conv_2_aggregated_features[0,15] + Conv_2_t[0,6]
   = 0.1640831530094147
 Conv_2_output_constraint[0,7]: 0.1483074277639389 Conv_1_Relu_ts[0,1]
   + 1.25136e-09 Conv_1_Relu_ts[0,2] - 3.02443e-09 Conv_1_Relu_ts[0,4]
   + 1.45591e-10 Conv_1_Relu_ts[0,5] - 3.6559e-05 Conv_1_Relu_ts[0,6]
   - 7.95116e-08 Conv_1_Relu_ts[0,7]
   + 0.0743252262473106 Conv_1_Relu_ts[0,9]
   + 2.19629e-08 Conv_1_Relu_ts[0,10] - 1.95369e-05 Conv_1_Relu_ts[0,11]
   - 1.21901e-08 Conv_1_Relu_ts[0,12]
   - 0.3415087759494781 Conv_1_Relu_ts[0,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[0,14]
   - 3.54178e-10 Conv_1_Relu_ts[0,15]
   + 0.2313222438097 Conv_2_aggregated_features[0,1]
   + 1.07351e-09 Conv_2_aggregated_features[0,2]
   - 4.05873e-09 Conv_2_aggregated_features[0,4]
   + 1.25187e-10 Conv_2_aggregated_features[0,5]
   - 3.00173e-05 Conv_2_aggregated_features[0,6]
   - 6.57874e-08 Conv_2_aggregated_features[0,7]
   + 0.0370698347687721 Conv_2_aggregated_features[0,9]
   - 2.2615e-08 Conv_2_aggregated_features[0,10]
   - 2.0281e-05 Conv_2_aggregated_features[0,11]
   - 1.14356e-08 Conv_2_aggregated_features[0,12]
   - 0.7083228826522827 Conv_2_aggregated_features[0,13]
   + 0.00042985 Conv_2_aggregated_features[0,14]
   - 4.8015e-10 Conv_2_aggregated_features[0,15] + Conv_2_t[0,7]
   = -0.5274350643157959
 Conv_2_output_constraint[0,8]: - 3.50628e-08 Conv_1_Relu_ts[0,1]
   - 0.00127368199173361 Conv_1_Relu_ts[0,5]
   + 1.1216e-10 Conv_1_Relu_ts[0,6] + 9.94055e-06 Conv_1_Relu_ts[0,9]
   + 2.4753e-05 Conv_1_Relu_ts[0,11] + 2.36679e-06 Conv_1_Relu_ts[0,13]
   + 1.90767e-09 Conv_1_Relu_ts[0,14] - 1.56108e-07 Conv_1_Relu_ts[0,15]
   - 5.18578e-06 Conv_2_aggregated_features[0,1]
   - 2.22254e-05 Conv_2_aggregated_features[0,5]
   + 9.48265e-11 Conv_2_aggregated_features[0,6]
   + 3.14595e-06 Conv_2_aggregated_features[0,9]
   - 1.16025e-05 Conv_2_aggregated_features[0,11]
   + 5.35568e-06 Conv_2_aggregated_features[0,13]
   + 1.47492e-09 Conv_2_aggregated_features[0,14]
   + 1.78245e-06 Conv_2_aggregated_features[0,15] + Conv_2_t[0,8]
   = 1.33952e-09
 Conv_2_output_constraint[0,9]: - 3.05926e-05 Conv_1_Relu_ts[0,1]
   - 4.69457e-10 Conv_1_Relu_ts[0,3] - 1.88994e-07 Conv_1_Relu_ts[0,5]
   + 1.14687e-08 Conv_1_Relu_ts[0,6] - 2.26795e-08 Conv_1_Relu_ts[0,7]
   - 1.1875e-05 Conv_1_Relu_ts[0,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[0,10]
   + 1.3279e-05 Conv_1_Relu_ts[0,11] - 8.03594e-13 Conv_1_Relu_ts[0,12]
   - 1.23268e-05 Conv_1_Relu_ts[0,13] - 1.4779e-07 Conv_1_Relu_ts[0,14]
   - 3.81296e-05 Conv_1_Relu_ts[0,15]
   + 3.05076e-11 Conv_2_aggregated_features[0,0]
   - 1.09971e-05 Conv_2_aggregated_features[0,1]
   - 6.1268e-09 Conv_2_aggregated_features[0,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[0,5]
   + 3.70939e-09 Conv_2_aggregated_features[0,6]
   - 2.02588e-06 Conv_2_aggregated_features[0,7]
   - 1.22094e-05 Conv_2_aggregated_features[0,9]
   - 1.30108e-06 Conv_2_aggregated_features[0,10]
   + 1.18736e-05 Conv_2_aggregated_features[0,11]
   + 9.38886e-13 Conv_2_aggregated_features[0,12]
   - 1.31367e-05 Conv_2_aggregated_features[0,13]
   + 1.5916e-09 Conv_2_aggregated_features[0,14]
   + 5.00944e-10 Conv_2_aggregated_features[0,15] + Conv_2_t[0,9]
   = 2.10696e-05
 Conv_2_output_constraint[0,10]: 2.28334e-09 Conv_1_Relu_ts[0,1]
   - 1.39061e-09 Conv_1_Relu_ts[0,6] + 1.19969e-08 Conv_1_Relu_ts[0,9]
   - 5.35695e-08 Conv_1_Relu_ts[0,10] + 7.265e-09 Conv_1_Relu_ts[0,13]
   - 2.17292e-11 Conv_1_Relu_ts[0,14] + 1.05271e-10 Conv_1_Relu_ts[0,15]
   + 4.90141e-09 Conv_2_aggregated_features[0,1]
   + 6.88e-06 Conv_2_aggregated_features[0,6]
   + 6.7107e-09 Conv_2_aggregated_features[0,9]
   + 1.08045e-08 Conv_2_aggregated_features[0,13]
   - 1.00381e-11 Conv_2_aggregated_features[0,14]
   - 5.27939e-06 Conv_2_aggregated_features[0,15] + Conv_2_t[0,10]
   = -6.36506e-09
 Conv_2_output_constraint[0,11]: - 9.57817e-09 Conv_1_Relu_ts[0,1]
   + 2.02195e-07 Conv_1_Relu_ts[0,2] - 3.79923e-06 Conv_1_Relu_ts[0,5]
   + 5.56008e-05 Conv_1_Relu_ts[0,6] - 5.0366e-09 Conv_1_Relu_ts[0,9]
   - 7.52285e-09 Conv_1_Relu_ts[0,13] - 5.6608e-10 Conv_1_Relu_ts[0,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[0,15]
   - 7.76309e-09 Conv_2_aggregated_features[0,1]
   - 2.62087e-08 Conv_2_aggregated_features[0,2]
   - 3.59639e-06 Conv_2_aggregated_features[0,5]
   - 2.75699e-10 Conv_2_aggregated_features[0,6]
   - 5.01117e-09 Conv_2_aggregated_features[0,9]
   - 7.1903e-09 Conv_2_aggregated_features[0,13]
   - 1.46623e-10 Conv_2_aggregated_features[0,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[0,15]
   + Conv_2_t[0,11] = 3.19541e-09
 Conv_2_output_constraint[0,12]: 1.20473e-05 Conv_1_Relu_ts[0,0]
   + 0.240223228931427 Conv_1_Relu_ts[0,1] + 1.328e-09 Conv_1_Relu_ts[0,2]
   + 2.75965e-08 Conv_1_Relu_ts[0,4] + 1.3558e-09 Conv_1_Relu_ts[0,5]
   - 1.34848e-06 Conv_1_Relu_ts[0,6] + 4.96526e-10 Conv_1_Relu_ts[0,7]
   - 0.3355287313461304 Conv_1_Relu_ts[0,9]
   + 2.31291e-07 Conv_1_Relu_ts[0,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[0,11]
   + 1.69037e-09 Conv_1_Relu_ts[0,12]
   - 0.1011878773570061 Conv_1_Relu_ts[0,13]
   - 0.0104798665270209 Conv_1_Relu_ts[0,14]
   - 4.11824e-11 Conv_1_Relu_ts[0,15]
   + 1.48158e-05 Conv_2_aggregated_features[0,0]
   - 1.069679498672485 Conv_2_aggregated_features[0,1]
   + 1.35995e-09 Conv_2_aggregated_features[0,2]
   + 2.54971e-08 Conv_2_aggregated_features[0,4]
   + 1.4479e-09 Conv_2_aggregated_features[0,5]
   - 2.57874e-05 Conv_2_aggregated_features[0,6]
   + 5.82765e-10 Conv_2_aggregated_features[0,7]
   + 0.2310869246721268 Conv_2_aggregated_features[0,9]
   + 2.42688e-07 Conv_2_aggregated_features[0,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[0,11]
   + 1.80254e-09 Conv_2_aggregated_features[0,12]
   + 0.1041423305869102 Conv_2_aggregated_features[0,13]
   - 0.00980580504983664 Conv_2_aggregated_features[0,14]
   + 4.55897e-11 Conv_2_aggregated_features[0,15] + Conv_2_t[0,12]
   = -0.0243597943335772
 Conv_2_output_constraint[0,13]: - 6.73028e-05 Conv_1_Relu_ts[0,1]
   + 6.15243e-11 Conv_1_Relu_ts[0,2] + 7.12143e-10 Conv_1_Relu_ts[0,4]
   - 3.05256e-08 Conv_1_Relu_ts[0,6]
   + 0.0179135110229254 Conv_1_Relu_ts[0,9]
   - 1.78315e-08 Conv_1_Relu_ts[0,10] + 3.95661e-08 Conv_1_Relu_ts[0,11]
   + 4.14048e-11 Conv_1_Relu_ts[0,12]
   - 0.00573352118954062 Conv_1_Relu_ts[0,13]
   - 1.5325e-05 Conv_1_Relu_ts[0,14]
   - 0.0551034025847912 Conv_2_aggregated_features[0,1]
   + 6.93423e-11 Conv_2_aggregated_features[0,2]
   + 2.18846e-10 Conv_2_aggregated_features[0,4]
   + 1.23153e-11 Conv_2_aggregated_features[0,5]
   - 9.5689e-08 Conv_2_aggregated_features[0,6]
   + 1.83226e-11 Conv_2_aggregated_features[0,7]
   + 0.0415055826306343 Conv_2_aggregated_features[0,9]
   - 4.74355e-09 Conv_2_aggregated_features[0,10]
   - 2.92725e-08 Conv_2_aggregated_features[0,11]
   + 3.37044e-11 Conv_2_aggregated_features[0,12]
   + 0.0228109527379274 Conv_2_aggregated_features[0,13]
   - 3.29092e-05 Conv_2_aggregated_features[0,14]
   - 2.71861e-12 Conv_2_aggregated_features[0,15] + Conv_2_t[0,13]
   = -0.0283597465604544
 Conv_2_output_constraint[0,14]: - 0.3400861024856567 Conv_1_Relu_ts[0,1]
   - 2.59888e-09 Conv_1_Relu_ts[0,2] + 1.10746e-09 Conv_1_Relu_ts[0,4]
   - 4.05832e-10 Conv_1_Relu_ts[0,5] - 2.28438e-05 Conv_1_Relu_ts[0,6]
   - 7.66288e-10 Conv_1_Relu_ts[0,7]
   - 0.1812356412410736 Conv_1_Relu_ts[0,9]
   + 2.36737e-07 Conv_1_Relu_ts[0,10] + 5.46304e-06 Conv_1_Relu_ts[0,11]
   - 1.05702e-08 Conv_1_Relu_ts[0,12]
   - 0.1597411930561066 Conv_1_Relu_ts[0,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[0,14]
   - 1.23647e-09 Conv_1_Relu_ts[0,15]
   - 0.0805128961801529 Conv_2_aggregated_features[0,1]
   - 2.67203e-09 Conv_2_aggregated_features[0,2]
   + 4.58926e-09 Conv_2_aggregated_features[0,4]
   - 5.32055e-10 Conv_2_aggregated_features[0,5]
   - 2.77383e-05 Conv_2_aggregated_features[0,6]
   - 5.76852e-10 Conv_2_aggregated_features[0,7]
   - 0.301887035369873 Conv_2_aggregated_features[0,9]
   + 2.91114e-07 Conv_2_aggregated_features[0,10]
   + 9.46222e-06 Conv_2_aggregated_features[0,11]
   - 1.05406e-08 Conv_2_aggregated_features[0,12]
   - 0.3912120759487152 Conv_2_aggregated_features[0,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[0,14]
   - 9.82623e-10 Conv_2_aggregated_features[0,15] + Conv_2_t[0,14]
   = 0.3463285267353058
 Conv_2_output_constraint[0,15]: 2.91582e-06 Conv_1_Relu_ts[0,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[0,2]
   - 6.96981e-13 Conv_1_Relu_ts[0,5] + 1.06195e-10 Conv_1_Relu_ts[0,6]
   + 9.12088e-06 Conv_1_Relu_ts[0,9] - 7.00879e-07 Conv_1_Relu_ts[0,10]
   + 2.54492e-08 Conv_1_Relu_ts[0,11] - 8.8874e-07 Conv_1_Relu_ts[0,12]
   - 9.8182e-06 Conv_1_Relu_ts[0,13] + 1.53315e-07 Conv_1_Relu_ts[0,14]
   - 2.41261e-08 Conv_1_Relu_ts[0,15]
   + 3.09046e-06 Conv_2_aggregated_features[0,1]
   - 5.23654e-07 Conv_2_aggregated_features[0,2]
   + 3.40881e-13 Conv_2_aggregated_features[0,5]
   + 4.84763e-12 Conv_2_aggregated_features[0,6]
   + 6.94651e-06 Conv_2_aggregated_features[0,9]
   + 1.80346e-08 Conv_2_aggregated_features[0,10]
   - 8.89515e-09 Conv_2_aggregated_features[0,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[0,12]
   - 5.75117e-06 Conv_2_aggregated_features[0,13]
   - 3.11447e-07 Conv_2_aggregated_features[0,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[0,15]
   + Conv_2_t[0,15] = -5.06887e-06
 Conv_2_output_constraint[1,0]: 5.62815e-10 Conv_1_Relu_ts[1,7]
   + 1.92775e-11 Conv_1_Relu_ts[1,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[1,15]
   + 5.32965e-11 Conv_2_aggregated_features[1,7]
   - 3.50224e-08 Conv_2_aggregated_features[1,15] + Conv_2_t[1,0]
   = 2.0494e-14
 Conv_2_output_constraint[1,1]: - 3.1209e-05 Conv_1_Relu_ts[1,1]
   + 3.2097e-12 Conv_1_Relu_ts[1,2] + 6.75828e-11 Conv_1_Relu_ts[1,4]
   + 6.05596e-11 Conv_1_Relu_ts[1,5] + 9.57194e-07 Conv_1_Relu_ts[1,6]
   - 1.94554e-13 Conv_1_Relu_ts[1,7] + 9.60207e-05 Conv_1_Relu_ts[1,9]
   + 3.54229e-11 Conv_1_Relu_ts[1,10] + 2.0034e-09 Conv_1_Relu_ts[1,11]
   + 3.41256e-05 Conv_1_Relu_ts[1,13] + 2.24541e-06 Conv_1_Relu_ts[1,14]
   - 7.56678e-10 Conv_1_Relu_ts[1,15]
   - 3.63258e-06 Conv_2_aggregated_features[1,1]
   - 2.02372e-12 Conv_2_aggregated_features[1,2]
   - 6.7717e-13 Conv_2_aggregated_features[1,4]
   + 1.64055e-10 Conv_2_aggregated_features[1,5]
   + 8.63851e-05 Conv_2_aggregated_features[1,6]
   - 5.43338e-10 Conv_2_aggregated_features[1,7]
   + 8.87988e-05 Conv_2_aggregated_features[1,9]
   + 2.03334e-11 Conv_2_aggregated_features[1,10]
   + 2.09865e-09 Conv_2_aggregated_features[1,11]
   + 5.1289e-05 Conv_2_aggregated_features[1,13]
   - 1.99215e-06 Conv_2_aggregated_features[1,14]
   - 2.85245e-06 Conv_2_aggregated_features[1,15] + Conv_2_t[1,1]
   = -5.09804e-05
 Conv_2_output_constraint[1,2]: 4.6554e-12 Conv_1_Relu_ts[1,1]
   - 5.08004e-07 Conv_1_Relu_ts[1,4] + 8.2088e-12 Conv_1_Relu_ts[1,9]
   + 3.89405e-05 Conv_1_Relu_ts[1,10] + 3.83811e-12 Conv_1_Relu_ts[1,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[1,15]
   + 2.16136e-12 Conv_2_aggregated_features[1,1]
   - 4.18879e-13 Conv_2_aggregated_features[1,4]
   + 1.01588e-11 Conv_2_aggregated_features[1,9]
   + 4.38408e-12 Conv_2_aggregated_features[1,13]
   + 6.90517e-06 Conv_2_aggregated_features[1,15] + Conv_2_t[1,2]
   = -1.21231e-11
 Conv_2_output_constraint[1,3]: - 9.77857e-07 Conv_1_Relu_ts[1,1]
   - 1.62341e-06 Conv_1_Relu_ts[1,4] - 1.26004e-09 Conv_1_Relu_ts[1,6]
   + 2.40003e-07 Conv_1_Relu_ts[1,7] - 5.41958e-07 Conv_1_Relu_ts[1,9]
   + 1.16363e-09 Conv_1_Relu_ts[1,10] - 2.13281e-11 Conv_1_Relu_ts[1,11]
   + 4.12955e-09 Conv_1_Relu_ts[1,12] - 3.71195e-07 Conv_1_Relu_ts[1,13]
   + 2.2241e-11 Conv_1_Relu_ts[1,14] + 6.37491e-09 Conv_1_Relu_ts[1,15]
   - 1.2731e-06 Conv_2_aggregated_features[1,1]
   - 1.0258e-06 Conv_2_aggregated_features[1,4]
   + 4.06095e-10 Conv_2_aggregated_features[1,6]
   + 3.12011e-07 Conv_2_aggregated_features[1,7]
   - 3.09535e-07 Conv_2_aggregated_features[1,9]
   - 0.00177046051248908 Conv_2_aggregated_features[1,10]
   + 8.54383e-13 Conv_2_aggregated_features[1,11]
   - 3.03582e-06 Conv_2_aggregated_features[1,12]
   - 4.41436e-07 Conv_2_aggregated_features[1,13]
   + 7.0584e-12 Conv_2_aggregated_features[1,14]
   - 9.96128e-05 Conv_2_aggregated_features[1,15] + Conv_2_t[1,3]
   = 4.02827e-07
 Conv_2_output_constraint[1,4]: 5.03273e-07 Conv_1_Relu_ts[1,1]
   - 6.22027e-09 Conv_1_Relu_ts[1,5] - 1.47914e-11 Conv_1_Relu_ts[1,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[1,9]
   + 1.18025e-13 Conv_1_Relu_ts[1,10] + 9.23352e-05 Conv_1_Relu_ts[1,11]
   + 0.000518991 Conv_1_Relu_ts[1,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[1,14]
   - 1.05921e-06 Conv_2_aggregated_features[1,1]
   + 2.88866e-10 Conv_2_aggregated_features[1,5]
   - 2.58968e-11 Conv_2_aggregated_features[1,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[1,9]
   - 3.93262e-05 Conv_2_aggregated_features[1,11]
   + 0.000690713 Conv_2_aggregated_features[1,13]
   + 7.70641e-05 Conv_2_aggregated_features[1,14] + Conv_2_t[1,4]
   = -5.84568e-07
 Conv_2_output_constraint[1,5]: 4.37405e-10 Conv_1_Relu_ts[1,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[1,4]
   + 1.77244e-09 Conv_1_Relu_ts[1,5] + 6.99595e-10 Conv_1_Relu_ts[1,9]
   - 1.77001e-13 Conv_1_Relu_ts[1,11] + 2.86208e-12 Conv_1_Relu_ts[1,12]
   + 3.24683e-10 Conv_1_Relu_ts[1,13] - 6.56311e-06 Conv_1_Relu_ts[1,15]
   - 4.63786e-09 Conv_2_aggregated_features[1,0]
   - 4.27797e-10 Conv_2_aggregated_features[1,1]
   - 2.50277e-13 Conv_2_aggregated_features[1,4]
   - 2.35888e-05 Conv_2_aggregated_features[1,5]
   + 6.12624e-10 Conv_2_aggregated_features[1,9]
   + 3.98546e-10 Conv_2_aggregated_features[1,13]
   + 9.89818e-10 Conv_2_aggregated_features[1,15] + Conv_2_t[1,5]
   = 6.24238e-10
 Conv_2_output_constraint[1,6]: - 4.80371e-09 Conv_1_Relu_ts[1,0]
   + 0.2580605149269104 Conv_1_Relu_ts[1,1]
   + 6.84534e-09 Conv_1_Relu_ts[1,2] + 7.44651e-09 Conv_1_Relu_ts[1,3]
   - 2.26729e-08 Conv_1_Relu_ts[1,4] + 1.08868e-09 Conv_1_Relu_ts[1,5]
   + 0.000151346 Conv_1_Relu_ts[1,6] - 1.07095e-09 Conv_1_Relu_ts[1,7]
   - 0.4001754522323608 Conv_1_Relu_ts[1,9]
   - 6.82816e-07 Conv_1_Relu_ts[1,10] - 3.7591e-05 Conv_1_Relu_ts[1,11]
   - 1.62843e-08 Conv_1_Relu_ts[1,12]
   + 1.447632431983948 Conv_1_Relu_ts[1,13]
   - 1.73427e-05 Conv_1_Relu_ts[1,14] + 2.28396e-09 Conv_1_Relu_ts[1,15]
   - 5.33024e-08 Conv_2_aggregated_features[1,0]
   + 0.8234332203865051 Conv_2_aggregated_features[1,1]
   + 6.68388e-09 Conv_2_aggregated_features[1,2]
   - 4.97866e-07 Conv_2_aggregated_features[1,3]
   - 2.61287e-08 Conv_2_aggregated_features[1,4]
   + 5.03108e-11 Conv_2_aggregated_features[1,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[1,6]
   - 1.69808e-09 Conv_2_aggregated_features[1,7]
   - 1.210363864898682 Conv_2_aggregated_features[1,9]
   - 7.75092e-07 Conv_2_aggregated_features[1,10]
   - 6.16735e-05 Conv_2_aggregated_features[1,11]
   - 1.9235e-08 Conv_2_aggregated_features[1,12]
   + 0.4373943507671356 Conv_2_aggregated_features[1,13]
   - 0.00871682073920965 Conv_2_aggregated_features[1,14]
   + 1.60146e-09 Conv_2_aggregated_features[1,15] + Conv_2_t[1,6]
   = 0.1640831530094147
 Conv_2_output_constraint[1,7]: 0.1483074277639389 Conv_1_Relu_ts[1,1]
   + 1.25136e-09 Conv_1_Relu_ts[1,2] - 3.02443e-09 Conv_1_Relu_ts[1,4]
   + 1.45591e-10 Conv_1_Relu_ts[1,5] - 3.6559e-05 Conv_1_Relu_ts[1,6]
   - 7.95116e-08 Conv_1_Relu_ts[1,7]
   + 0.0743252262473106 Conv_1_Relu_ts[1,9]
   + 2.19629e-08 Conv_1_Relu_ts[1,10] - 1.95369e-05 Conv_1_Relu_ts[1,11]
   - 1.21901e-08 Conv_1_Relu_ts[1,12]
   - 0.3415087759494781 Conv_1_Relu_ts[1,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[1,14]
   - 3.54178e-10 Conv_1_Relu_ts[1,15]
   + 0.2313222438097 Conv_2_aggregated_features[1,1]
   + 1.07351e-09 Conv_2_aggregated_features[1,2]
   - 4.05873e-09 Conv_2_aggregated_features[1,4]
   + 1.25187e-10 Conv_2_aggregated_features[1,5]
   - 3.00173e-05 Conv_2_aggregated_features[1,6]
   - 6.57874e-08 Conv_2_aggregated_features[1,7]
   + 0.0370698347687721 Conv_2_aggregated_features[1,9]
   - 2.2615e-08 Conv_2_aggregated_features[1,10]
   - 2.0281e-05 Conv_2_aggregated_features[1,11]
   - 1.14356e-08 Conv_2_aggregated_features[1,12]
   - 0.7083228826522827 Conv_2_aggregated_features[1,13]
   + 0.00042985 Conv_2_aggregated_features[1,14]
   - 4.8015e-10 Conv_2_aggregated_features[1,15] + Conv_2_t[1,7]
   = -0.5274350643157959
 Conv_2_output_constraint[1,8]: - 3.50628e-08 Conv_1_Relu_ts[1,1]
   - 0.00127368199173361 Conv_1_Relu_ts[1,5]
   + 1.1216e-10 Conv_1_Relu_ts[1,6] + 9.94055e-06 Conv_1_Relu_ts[1,9]
   + 2.4753e-05 Conv_1_Relu_ts[1,11] + 2.36679e-06 Conv_1_Relu_ts[1,13]
   + 1.90767e-09 Conv_1_Relu_ts[1,14] - 1.56108e-07 Conv_1_Relu_ts[1,15]
   - 5.18578e-06 Conv_2_aggregated_features[1,1]
   - 2.22254e-05 Conv_2_aggregated_features[1,5]
   + 9.48265e-11 Conv_2_aggregated_features[1,6]
   + 3.14595e-06 Conv_2_aggregated_features[1,9]
   - 1.16025e-05 Conv_2_aggregated_features[1,11]
   + 5.35568e-06 Conv_2_aggregated_features[1,13]
   + 1.47492e-09 Conv_2_aggregated_features[1,14]
   + 1.78245e-06 Conv_2_aggregated_features[1,15] + Conv_2_t[1,8]
   = 1.33952e-09
 Conv_2_output_constraint[1,9]: - 3.05926e-05 Conv_1_Relu_ts[1,1]
   - 4.69457e-10 Conv_1_Relu_ts[1,3] - 1.88994e-07 Conv_1_Relu_ts[1,5]
   + 1.14687e-08 Conv_1_Relu_ts[1,6] - 2.26795e-08 Conv_1_Relu_ts[1,7]
   - 1.1875e-05 Conv_1_Relu_ts[1,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[1,10]
   + 1.3279e-05 Conv_1_Relu_ts[1,11] - 8.03594e-13 Conv_1_Relu_ts[1,12]
   - 1.23268e-05 Conv_1_Relu_ts[1,13] - 1.4779e-07 Conv_1_Relu_ts[1,14]
   - 3.81296e-05 Conv_1_Relu_ts[1,15]
   + 3.05076e-11 Conv_2_aggregated_features[1,0]
   - 1.09971e-05 Conv_2_aggregated_features[1,1]
   - 6.1268e-09 Conv_2_aggregated_features[1,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[1,5]
   + 3.70939e-09 Conv_2_aggregated_features[1,6]
   - 2.02588e-06 Conv_2_aggregated_features[1,7]
   - 1.22094e-05 Conv_2_aggregated_features[1,9]
   - 1.30108e-06 Conv_2_aggregated_features[1,10]
   + 1.18736e-05 Conv_2_aggregated_features[1,11]
   + 9.38886e-13 Conv_2_aggregated_features[1,12]
   - 1.31367e-05 Conv_2_aggregated_features[1,13]
   + 1.5916e-09 Conv_2_aggregated_features[1,14]
   + 5.00944e-10 Conv_2_aggregated_features[1,15] + Conv_2_t[1,9]
   = 2.10696e-05
 Conv_2_output_constraint[1,10]: 2.28334e-09 Conv_1_Relu_ts[1,1]
   - 1.39061e-09 Conv_1_Relu_ts[1,6] + 1.19969e-08 Conv_1_Relu_ts[1,9]
   - 5.35695e-08 Conv_1_Relu_ts[1,10] + 7.265e-09 Conv_1_Relu_ts[1,13]
   - 2.17292e-11 Conv_1_Relu_ts[1,14] + 1.05271e-10 Conv_1_Relu_ts[1,15]
   + 4.90141e-09 Conv_2_aggregated_features[1,1]
   + 6.88e-06 Conv_2_aggregated_features[1,6]
   + 6.7107e-09 Conv_2_aggregated_features[1,9]
   + 1.08045e-08 Conv_2_aggregated_features[1,13]
   - 1.00381e-11 Conv_2_aggregated_features[1,14]
   - 5.27939e-06 Conv_2_aggregated_features[1,15] + Conv_2_t[1,10]
   = -6.36506e-09
 Conv_2_output_constraint[1,11]: - 9.57817e-09 Conv_1_Relu_ts[1,1]
   + 2.02195e-07 Conv_1_Relu_ts[1,2] - 3.79923e-06 Conv_1_Relu_ts[1,5]
   + 5.56008e-05 Conv_1_Relu_ts[1,6] - 5.0366e-09 Conv_1_Relu_ts[1,9]
   - 7.52285e-09 Conv_1_Relu_ts[1,13] - 5.6608e-10 Conv_1_Relu_ts[1,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[1,15]
   - 7.76309e-09 Conv_2_aggregated_features[1,1]
   - 2.62087e-08 Conv_2_aggregated_features[1,2]
   - 3.59639e-06 Conv_2_aggregated_features[1,5]
   - 2.75699e-10 Conv_2_aggregated_features[1,6]
   - 5.01117e-09 Conv_2_aggregated_features[1,9]
   - 7.1903e-09 Conv_2_aggregated_features[1,13]
   - 1.46623e-10 Conv_2_aggregated_features[1,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[1,15]
   + Conv_2_t[1,11] = 3.19541e-09
 Conv_2_output_constraint[1,12]: 1.20473e-05 Conv_1_Relu_ts[1,0]
   + 0.240223228931427 Conv_1_Relu_ts[1,1] + 1.328e-09 Conv_1_Relu_ts[1,2]
   + 2.75965e-08 Conv_1_Relu_ts[1,4] + 1.3558e-09 Conv_1_Relu_ts[1,5]
   - 1.34848e-06 Conv_1_Relu_ts[1,6] + 4.96526e-10 Conv_1_Relu_ts[1,7]
   - 0.3355287313461304 Conv_1_Relu_ts[1,9]
   + 2.31291e-07 Conv_1_Relu_ts[1,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[1,11]
   + 1.69037e-09 Conv_1_Relu_ts[1,12]
   - 0.1011878773570061 Conv_1_Relu_ts[1,13]
   - 0.0104798665270209 Conv_1_Relu_ts[1,14]
   - 4.11824e-11 Conv_1_Relu_ts[1,15]
   + 1.48158e-05 Conv_2_aggregated_features[1,0]
   - 1.069679498672485 Conv_2_aggregated_features[1,1]
   + 1.35995e-09 Conv_2_aggregated_features[1,2]
   + 2.54971e-08 Conv_2_aggregated_features[1,4]
   + 1.4479e-09 Conv_2_aggregated_features[1,5]
   - 2.57874e-05 Conv_2_aggregated_features[1,6]
   + 5.82765e-10 Conv_2_aggregated_features[1,7]
   + 0.2310869246721268 Conv_2_aggregated_features[1,9]
   + 2.42688e-07 Conv_2_aggregated_features[1,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[1,11]
   + 1.80254e-09 Conv_2_aggregated_features[1,12]
   + 0.1041423305869102 Conv_2_aggregated_features[1,13]
   - 0.00980580504983664 Conv_2_aggregated_features[1,14]
   + 4.55897e-11 Conv_2_aggregated_features[1,15] + Conv_2_t[1,12]
   = -0.0243597943335772
 Conv_2_output_constraint[1,13]: - 6.73028e-05 Conv_1_Relu_ts[1,1]
   + 6.15243e-11 Conv_1_Relu_ts[1,2] + 7.12143e-10 Conv_1_Relu_ts[1,4]
   - 3.05256e-08 Conv_1_Relu_ts[1,6]
   + 0.0179135110229254 Conv_1_Relu_ts[1,9]
   - 1.78315e-08 Conv_1_Relu_ts[1,10] + 3.95661e-08 Conv_1_Relu_ts[1,11]
   + 4.14048e-11 Conv_1_Relu_ts[1,12]
   - 0.00573352118954062 Conv_1_Relu_ts[1,13]
   - 1.5325e-05 Conv_1_Relu_ts[1,14]
   - 0.0551034025847912 Conv_2_aggregated_features[1,1]
   + 6.93423e-11 Conv_2_aggregated_features[1,2]
   + 2.18846e-10 Conv_2_aggregated_features[1,4]
   + 1.23153e-11 Conv_2_aggregated_features[1,5]
   - 9.5689e-08 Conv_2_aggregated_features[1,6]
   + 1.83226e-11 Conv_2_aggregated_features[1,7]
   + 0.0415055826306343 Conv_2_aggregated_features[1,9]
   - 4.74355e-09 Conv_2_aggregated_features[1,10]
   - 2.92725e-08 Conv_2_aggregated_features[1,11]
   + 3.37044e-11 Conv_2_aggregated_features[1,12]
   + 0.0228109527379274 Conv_2_aggregated_features[1,13]
   - 3.29092e-05 Conv_2_aggregated_features[1,14]
   - 2.71861e-12 Conv_2_aggregated_features[1,15] + Conv_2_t[1,13]
   = -0.0283597465604544
 Conv_2_output_constraint[1,14]: - 0.3400861024856567 Conv_1_Relu_ts[1,1]
   - 2.59888e-09 Conv_1_Relu_ts[1,2] + 1.10746e-09 Conv_1_Relu_ts[1,4]
   - 4.05832e-10 Conv_1_Relu_ts[1,5] - 2.28438e-05 Conv_1_Relu_ts[1,6]
   - 7.66288e-10 Conv_1_Relu_ts[1,7]
   - 0.1812356412410736 Conv_1_Relu_ts[1,9]
   + 2.36737e-07 Conv_1_Relu_ts[1,10] + 5.46304e-06 Conv_1_Relu_ts[1,11]
   - 1.05702e-08 Conv_1_Relu_ts[1,12]
   - 0.1597411930561066 Conv_1_Relu_ts[1,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[1,14]
   - 1.23647e-09 Conv_1_Relu_ts[1,15]
   - 0.0805128961801529 Conv_2_aggregated_features[1,1]
   - 2.67203e-09 Conv_2_aggregated_features[1,2]
   + 4.58926e-09 Conv_2_aggregated_features[1,4]
   - 5.32055e-10 Conv_2_aggregated_features[1,5]
   - 2.77383e-05 Conv_2_aggregated_features[1,6]
   - 5.76852e-10 Conv_2_aggregated_features[1,7]
   - 0.301887035369873 Conv_2_aggregated_features[1,9]
   + 2.91114e-07 Conv_2_aggregated_features[1,10]
   + 9.46222e-06 Conv_2_aggregated_features[1,11]
   - 1.05406e-08 Conv_2_aggregated_features[1,12]
   - 0.3912120759487152 Conv_2_aggregated_features[1,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[1,14]
   - 9.82623e-10 Conv_2_aggregated_features[1,15] + Conv_2_t[1,14]
   = 0.3463285267353058
 Conv_2_output_constraint[1,15]: 2.91582e-06 Conv_1_Relu_ts[1,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[1,2]
   - 6.96981e-13 Conv_1_Relu_ts[1,5] + 1.06195e-10 Conv_1_Relu_ts[1,6]
   + 9.12088e-06 Conv_1_Relu_ts[1,9] - 7.00879e-07 Conv_1_Relu_ts[1,10]
   + 2.54492e-08 Conv_1_Relu_ts[1,11] - 8.8874e-07 Conv_1_Relu_ts[1,12]
   - 9.8182e-06 Conv_1_Relu_ts[1,13] + 1.53315e-07 Conv_1_Relu_ts[1,14]
   - 2.41261e-08 Conv_1_Relu_ts[1,15]
   + 3.09046e-06 Conv_2_aggregated_features[1,1]
   - 5.23654e-07 Conv_2_aggregated_features[1,2]
   + 3.40881e-13 Conv_2_aggregated_features[1,5]
   + 4.84763e-12 Conv_2_aggregated_features[1,6]
   + 6.94651e-06 Conv_2_aggregated_features[1,9]
   + 1.80346e-08 Conv_2_aggregated_features[1,10]
   - 8.89515e-09 Conv_2_aggregated_features[1,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[1,12]
   - 5.75117e-06 Conv_2_aggregated_features[1,13]
   - 3.11447e-07 Conv_2_aggregated_features[1,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[1,15]
   + Conv_2_t[1,15] = -5.06887e-06
 Conv_2_output_constraint[2,0]: 5.62815e-10 Conv_1_Relu_ts[2,7]
   + 1.92775e-11 Conv_1_Relu_ts[2,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[2,15]
   + 5.32965e-11 Conv_2_aggregated_features[2,7]
   - 3.50224e-08 Conv_2_aggregated_features[2,15] + Conv_2_t[2,0]
   = 2.0494e-14
 Conv_2_output_constraint[2,1]: - 3.1209e-05 Conv_1_Relu_ts[2,1]
   + 3.2097e-12 Conv_1_Relu_ts[2,2] + 6.75828e-11 Conv_1_Relu_ts[2,4]
   + 6.05596e-11 Conv_1_Relu_ts[2,5] + 9.57194e-07 Conv_1_Relu_ts[2,6]
   - 1.94554e-13 Conv_1_Relu_ts[2,7] + 9.60207e-05 Conv_1_Relu_ts[2,9]
   + 3.54229e-11 Conv_1_Relu_ts[2,10] + 2.0034e-09 Conv_1_Relu_ts[2,11]
   + 3.41256e-05 Conv_1_Relu_ts[2,13] + 2.24541e-06 Conv_1_Relu_ts[2,14]
   - 7.56678e-10 Conv_1_Relu_ts[2,15]
   - 3.63258e-06 Conv_2_aggregated_features[2,1]
   - 2.02372e-12 Conv_2_aggregated_features[2,2]
   - 6.7717e-13 Conv_2_aggregated_features[2,4]
   + 1.64055e-10 Conv_2_aggregated_features[2,5]
   + 8.63851e-05 Conv_2_aggregated_features[2,6]
   - 5.43338e-10 Conv_2_aggregated_features[2,7]
   + 8.87988e-05 Conv_2_aggregated_features[2,9]
   + 2.03334e-11 Conv_2_aggregated_features[2,10]
   + 2.09865e-09 Conv_2_aggregated_features[2,11]
   + 5.1289e-05 Conv_2_aggregated_features[2,13]
   - 1.99215e-06 Conv_2_aggregated_features[2,14]
   - 2.85245e-06 Conv_2_aggregated_features[2,15] + Conv_2_t[2,1]
   = -5.09804e-05
 Conv_2_output_constraint[2,2]: 4.6554e-12 Conv_1_Relu_ts[2,1]
   - 5.08004e-07 Conv_1_Relu_ts[2,4] + 8.2088e-12 Conv_1_Relu_ts[2,9]
   + 3.89405e-05 Conv_1_Relu_ts[2,10] + 3.83811e-12 Conv_1_Relu_ts[2,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[2,15]
   + 2.16136e-12 Conv_2_aggregated_features[2,1]
   - 4.18879e-13 Conv_2_aggregated_features[2,4]
   + 1.01588e-11 Conv_2_aggregated_features[2,9]
   + 4.38408e-12 Conv_2_aggregated_features[2,13]
   + 6.90517e-06 Conv_2_aggregated_features[2,15] + Conv_2_t[2,2]
   = -1.21231e-11
 Conv_2_output_constraint[2,3]: - 9.77857e-07 Conv_1_Relu_ts[2,1]
   - 1.62341e-06 Conv_1_Relu_ts[2,4] - 1.26004e-09 Conv_1_Relu_ts[2,6]
   + 2.40003e-07 Conv_1_Relu_ts[2,7] - 5.41958e-07 Conv_1_Relu_ts[2,9]
   + 1.16363e-09 Conv_1_Relu_ts[2,10] - 2.13281e-11 Conv_1_Relu_ts[2,11]
   + 4.12955e-09 Conv_1_Relu_ts[2,12] - 3.71195e-07 Conv_1_Relu_ts[2,13]
   + 2.2241e-11 Conv_1_Relu_ts[2,14] + 6.37491e-09 Conv_1_Relu_ts[2,15]
   - 1.2731e-06 Conv_2_aggregated_features[2,1]
   - 1.0258e-06 Conv_2_aggregated_features[2,4]
   + 4.06095e-10 Conv_2_aggregated_features[2,6]
   + 3.12011e-07 Conv_2_aggregated_features[2,7]
   - 3.09535e-07 Conv_2_aggregated_features[2,9]
   - 0.00177046051248908 Conv_2_aggregated_features[2,10]
   + 8.54383e-13 Conv_2_aggregated_features[2,11]
   - 3.03582e-06 Conv_2_aggregated_features[2,12]
   - 4.41436e-07 Conv_2_aggregated_features[2,13]
   + 7.0584e-12 Conv_2_aggregated_features[2,14]
   - 9.96128e-05 Conv_2_aggregated_features[2,15] + Conv_2_t[2,3]
   = 4.02827e-07
 Conv_2_output_constraint[2,4]: 5.03273e-07 Conv_1_Relu_ts[2,1]
   - 6.22027e-09 Conv_1_Relu_ts[2,5] - 1.47914e-11 Conv_1_Relu_ts[2,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[2,9]
   + 1.18025e-13 Conv_1_Relu_ts[2,10] + 9.23352e-05 Conv_1_Relu_ts[2,11]
   + 0.000518991 Conv_1_Relu_ts[2,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[2,14]
   - 1.05921e-06 Conv_2_aggregated_features[2,1]
   + 2.88866e-10 Conv_2_aggregated_features[2,5]
   - 2.58968e-11 Conv_2_aggregated_features[2,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[2,9]
   - 3.93262e-05 Conv_2_aggregated_features[2,11]
   + 0.000690713 Conv_2_aggregated_features[2,13]
   + 7.70641e-05 Conv_2_aggregated_features[2,14] + Conv_2_t[2,4]
   = -5.84568e-07
 Conv_2_output_constraint[2,5]: 4.37405e-10 Conv_1_Relu_ts[2,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[2,4]
   + 1.77244e-09 Conv_1_Relu_ts[2,5] + 6.99595e-10 Conv_1_Relu_ts[2,9]
   - 1.77001e-13 Conv_1_Relu_ts[2,11] + 2.86208e-12 Conv_1_Relu_ts[2,12]
   + 3.24683e-10 Conv_1_Relu_ts[2,13] - 6.56311e-06 Conv_1_Relu_ts[2,15]
   - 4.63786e-09 Conv_2_aggregated_features[2,0]
   - 4.27797e-10 Conv_2_aggregated_features[2,1]
   - 2.50277e-13 Conv_2_aggregated_features[2,4]
   - 2.35888e-05 Conv_2_aggregated_features[2,5]
   + 6.12624e-10 Conv_2_aggregated_features[2,9]
   + 3.98546e-10 Conv_2_aggregated_features[2,13]
   + 9.89818e-10 Conv_2_aggregated_features[2,15] + Conv_2_t[2,5]
   = 6.24238e-10
 Conv_2_output_constraint[2,6]: - 4.80371e-09 Conv_1_Relu_ts[2,0]
   + 0.2580605149269104 Conv_1_Relu_ts[2,1]
   + 6.84534e-09 Conv_1_Relu_ts[2,2] + 7.44651e-09 Conv_1_Relu_ts[2,3]
   - 2.26729e-08 Conv_1_Relu_ts[2,4] + 1.08868e-09 Conv_1_Relu_ts[2,5]
   + 0.000151346 Conv_1_Relu_ts[2,6] - 1.07095e-09 Conv_1_Relu_ts[2,7]
   - 0.4001754522323608 Conv_1_Relu_ts[2,9]
   - 6.82816e-07 Conv_1_Relu_ts[2,10] - 3.7591e-05 Conv_1_Relu_ts[2,11]
   - 1.62843e-08 Conv_1_Relu_ts[2,12]
   + 1.447632431983948 Conv_1_Relu_ts[2,13]
   - 1.73427e-05 Conv_1_Relu_ts[2,14] + 2.28396e-09 Conv_1_Relu_ts[2,15]
   - 5.33024e-08 Conv_2_aggregated_features[2,0]
   + 0.8234332203865051 Conv_2_aggregated_features[2,1]
   + 6.68388e-09 Conv_2_aggregated_features[2,2]
   - 4.97866e-07 Conv_2_aggregated_features[2,3]
   - 2.61287e-08 Conv_2_aggregated_features[2,4]
   + 5.03108e-11 Conv_2_aggregated_features[2,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[2,6]
   - 1.69808e-09 Conv_2_aggregated_features[2,7]
   - 1.210363864898682 Conv_2_aggregated_features[2,9]
   - 7.75092e-07 Conv_2_aggregated_features[2,10]
   - 6.16735e-05 Conv_2_aggregated_features[2,11]
   - 1.9235e-08 Conv_2_aggregated_features[2,12]
   + 0.4373943507671356 Conv_2_aggregated_features[2,13]
   - 0.00871682073920965 Conv_2_aggregated_features[2,14]
   + 1.60146e-09 Conv_2_aggregated_features[2,15] + Conv_2_t[2,6]
   = 0.1640831530094147
 Conv_2_output_constraint[2,7]: 0.1483074277639389 Conv_1_Relu_ts[2,1]
   + 1.25136e-09 Conv_1_Relu_ts[2,2] - 3.02443e-09 Conv_1_Relu_ts[2,4]
   + 1.45591e-10 Conv_1_Relu_ts[2,5] - 3.6559e-05 Conv_1_Relu_ts[2,6]
   - 7.95116e-08 Conv_1_Relu_ts[2,7]
   + 0.0743252262473106 Conv_1_Relu_ts[2,9]
   + 2.19629e-08 Conv_1_Relu_ts[2,10] - 1.95369e-05 Conv_1_Relu_ts[2,11]
   - 1.21901e-08 Conv_1_Relu_ts[2,12]
   - 0.3415087759494781 Conv_1_Relu_ts[2,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[2,14]
   - 3.54178e-10 Conv_1_Relu_ts[2,15]
   + 0.2313222438097 Conv_2_aggregated_features[2,1]
   + 1.07351e-09 Conv_2_aggregated_features[2,2]
   - 4.05873e-09 Conv_2_aggregated_features[2,4]
   + 1.25187e-10 Conv_2_aggregated_features[2,5]
   - 3.00173e-05 Conv_2_aggregated_features[2,6]
   - 6.57874e-08 Conv_2_aggregated_features[2,7]
   + 0.0370698347687721 Conv_2_aggregated_features[2,9]
   - 2.2615e-08 Conv_2_aggregated_features[2,10]
   - 2.0281e-05 Conv_2_aggregated_features[2,11]
   - 1.14356e-08 Conv_2_aggregated_features[2,12]
   - 0.7083228826522827 Conv_2_aggregated_features[2,13]
   + 0.00042985 Conv_2_aggregated_features[2,14]
   - 4.8015e-10 Conv_2_aggregated_features[2,15] + Conv_2_t[2,7]
   = -0.5274350643157959
 Conv_2_output_constraint[2,8]: - 3.50628e-08 Conv_1_Relu_ts[2,1]
   - 0.00127368199173361 Conv_1_Relu_ts[2,5]
   + 1.1216e-10 Conv_1_Relu_ts[2,6] + 9.94055e-06 Conv_1_Relu_ts[2,9]
   + 2.4753e-05 Conv_1_Relu_ts[2,11] + 2.36679e-06 Conv_1_Relu_ts[2,13]
   + 1.90767e-09 Conv_1_Relu_ts[2,14] - 1.56108e-07 Conv_1_Relu_ts[2,15]
   - 5.18578e-06 Conv_2_aggregated_features[2,1]
   - 2.22254e-05 Conv_2_aggregated_features[2,5]
   + 9.48265e-11 Conv_2_aggregated_features[2,6]
   + 3.14595e-06 Conv_2_aggregated_features[2,9]
   - 1.16025e-05 Conv_2_aggregated_features[2,11]
   + 5.35568e-06 Conv_2_aggregated_features[2,13]
   + 1.47492e-09 Conv_2_aggregated_features[2,14]
   + 1.78245e-06 Conv_2_aggregated_features[2,15] + Conv_2_t[2,8]
   = 1.33952e-09
 Conv_2_output_constraint[2,9]: - 3.05926e-05 Conv_1_Relu_ts[2,1]
   - 4.69457e-10 Conv_1_Relu_ts[2,3] - 1.88994e-07 Conv_1_Relu_ts[2,5]
   + 1.14687e-08 Conv_1_Relu_ts[2,6] - 2.26795e-08 Conv_1_Relu_ts[2,7]
   - 1.1875e-05 Conv_1_Relu_ts[2,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[2,10]
   + 1.3279e-05 Conv_1_Relu_ts[2,11] - 8.03594e-13 Conv_1_Relu_ts[2,12]
   - 1.23268e-05 Conv_1_Relu_ts[2,13] - 1.4779e-07 Conv_1_Relu_ts[2,14]
   - 3.81296e-05 Conv_1_Relu_ts[2,15]
   + 3.05076e-11 Conv_2_aggregated_features[2,0]
   - 1.09971e-05 Conv_2_aggregated_features[2,1]
   - 6.1268e-09 Conv_2_aggregated_features[2,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[2,5]
   + 3.70939e-09 Conv_2_aggregated_features[2,6]
   - 2.02588e-06 Conv_2_aggregated_features[2,7]
   - 1.22094e-05 Conv_2_aggregated_features[2,9]
   - 1.30108e-06 Conv_2_aggregated_features[2,10]
   + 1.18736e-05 Conv_2_aggregated_features[2,11]
   + 9.38886e-13 Conv_2_aggregated_features[2,12]
   - 1.31367e-05 Conv_2_aggregated_features[2,13]
   + 1.5916e-09 Conv_2_aggregated_features[2,14]
   + 5.00944e-10 Conv_2_aggregated_features[2,15] + Conv_2_t[2,9]
   = 2.10696e-05
 Conv_2_output_constraint[2,10]: 2.28334e-09 Conv_1_Relu_ts[2,1]
   - 1.39061e-09 Conv_1_Relu_ts[2,6] + 1.19969e-08 Conv_1_Relu_ts[2,9]
   - 5.35695e-08 Conv_1_Relu_ts[2,10] + 7.265e-09 Conv_1_Relu_ts[2,13]
   - 2.17292e-11 Conv_1_Relu_ts[2,14] + 1.05271e-10 Conv_1_Relu_ts[2,15]
   + 4.90141e-09 Conv_2_aggregated_features[2,1]
   + 6.88e-06 Conv_2_aggregated_features[2,6]
   + 6.7107e-09 Conv_2_aggregated_features[2,9]
   + 1.08045e-08 Conv_2_aggregated_features[2,13]
   - 1.00381e-11 Conv_2_aggregated_features[2,14]
   - 5.27939e-06 Conv_2_aggregated_features[2,15] + Conv_2_t[2,10]
   = -6.36506e-09
 Conv_2_output_constraint[2,11]: - 9.57817e-09 Conv_1_Relu_ts[2,1]
   + 2.02195e-07 Conv_1_Relu_ts[2,2] - 3.79923e-06 Conv_1_Relu_ts[2,5]
   + 5.56008e-05 Conv_1_Relu_ts[2,6] - 5.0366e-09 Conv_1_Relu_ts[2,9]
   - 7.52285e-09 Conv_1_Relu_ts[2,13] - 5.6608e-10 Conv_1_Relu_ts[2,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[2,15]
   - 7.76309e-09 Conv_2_aggregated_features[2,1]
   - 2.62087e-08 Conv_2_aggregated_features[2,2]
   - 3.59639e-06 Conv_2_aggregated_features[2,5]
   - 2.75699e-10 Conv_2_aggregated_features[2,6]
   - 5.01117e-09 Conv_2_aggregated_features[2,9]
   - 7.1903e-09 Conv_2_aggregated_features[2,13]
   - 1.46623e-10 Conv_2_aggregated_features[2,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[2,15]
   + Conv_2_t[2,11] = 3.19541e-09
 Conv_2_output_constraint[2,12]: 1.20473e-05 Conv_1_Relu_ts[2,0]
   + 0.240223228931427 Conv_1_Relu_ts[2,1] + 1.328e-09 Conv_1_Relu_ts[2,2]
   + 2.75965e-08 Conv_1_Relu_ts[2,4] + 1.3558e-09 Conv_1_Relu_ts[2,5]
   - 1.34848e-06 Conv_1_Relu_ts[2,6] + 4.96526e-10 Conv_1_Relu_ts[2,7]
   - 0.3355287313461304 Conv_1_Relu_ts[2,9]
   + 2.31291e-07 Conv_1_Relu_ts[2,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[2,11]
   + 1.69037e-09 Conv_1_Relu_ts[2,12]
   - 0.1011878773570061 Conv_1_Relu_ts[2,13]
   - 0.0104798665270209 Conv_1_Relu_ts[2,14]
   - 4.11824e-11 Conv_1_Relu_ts[2,15]
   + 1.48158e-05 Conv_2_aggregated_features[2,0]
   - 1.069679498672485 Conv_2_aggregated_features[2,1]
   + 1.35995e-09 Conv_2_aggregated_features[2,2]
   + 2.54971e-08 Conv_2_aggregated_features[2,4]
   + 1.4479e-09 Conv_2_aggregated_features[2,5]
   - 2.57874e-05 Conv_2_aggregated_features[2,6]
   + 5.82765e-10 Conv_2_aggregated_features[2,7]
   + 0.2310869246721268 Conv_2_aggregated_features[2,9]
   + 2.42688e-07 Conv_2_aggregated_features[2,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[2,11]
   + 1.80254e-09 Conv_2_aggregated_features[2,12]
   + 0.1041423305869102 Conv_2_aggregated_features[2,13]
   - 0.00980580504983664 Conv_2_aggregated_features[2,14]
   + 4.55897e-11 Conv_2_aggregated_features[2,15] + Conv_2_t[2,12]
   = -0.0243597943335772
 Conv_2_output_constraint[2,13]: - 6.73028e-05 Conv_1_Relu_ts[2,1]
   + 6.15243e-11 Conv_1_Relu_ts[2,2] + 7.12143e-10 Conv_1_Relu_ts[2,4]
   - 3.05256e-08 Conv_1_Relu_ts[2,6]
   + 0.0179135110229254 Conv_1_Relu_ts[2,9]
   - 1.78315e-08 Conv_1_Relu_ts[2,10] + 3.95661e-08 Conv_1_Relu_ts[2,11]
   + 4.14048e-11 Conv_1_Relu_ts[2,12]
   - 0.00573352118954062 Conv_1_Relu_ts[2,13]
   - 1.5325e-05 Conv_1_Relu_ts[2,14]
   - 0.0551034025847912 Conv_2_aggregated_features[2,1]
   + 6.93423e-11 Conv_2_aggregated_features[2,2]
   + 2.18846e-10 Conv_2_aggregated_features[2,4]
   + 1.23153e-11 Conv_2_aggregated_features[2,5]
   - 9.5689e-08 Conv_2_aggregated_features[2,6]
   + 1.83226e-11 Conv_2_aggregated_features[2,7]
   + 0.0415055826306343 Conv_2_aggregated_features[2,9]
   - 4.74355e-09 Conv_2_aggregated_features[2,10]
   - 2.92725e-08 Conv_2_aggregated_features[2,11]
   + 3.37044e-11 Conv_2_aggregated_features[2,12]
   + 0.0228109527379274 Conv_2_aggregated_features[2,13]
   - 3.29092e-05 Conv_2_aggregated_features[2,14]
   - 2.71861e-12 Conv_2_aggregated_features[2,15] + Conv_2_t[2,13]
   = -0.0283597465604544
 Conv_2_output_constraint[2,14]: - 0.3400861024856567 Conv_1_Relu_ts[2,1]
   - 2.59888e-09 Conv_1_Relu_ts[2,2] + 1.10746e-09 Conv_1_Relu_ts[2,4]
   - 4.05832e-10 Conv_1_Relu_ts[2,5] - 2.28438e-05 Conv_1_Relu_ts[2,6]
   - 7.66288e-10 Conv_1_Relu_ts[2,7]
   - 0.1812356412410736 Conv_1_Relu_ts[2,9]
   + 2.36737e-07 Conv_1_Relu_ts[2,10] + 5.46304e-06 Conv_1_Relu_ts[2,11]
   - 1.05702e-08 Conv_1_Relu_ts[2,12]
   - 0.1597411930561066 Conv_1_Relu_ts[2,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[2,14]
   - 1.23647e-09 Conv_1_Relu_ts[2,15]
   - 0.0805128961801529 Conv_2_aggregated_features[2,1]
   - 2.67203e-09 Conv_2_aggregated_features[2,2]
   + 4.58926e-09 Conv_2_aggregated_features[2,4]
   - 5.32055e-10 Conv_2_aggregated_features[2,5]
   - 2.77383e-05 Conv_2_aggregated_features[2,6]
   - 5.76852e-10 Conv_2_aggregated_features[2,7]
   - 0.301887035369873 Conv_2_aggregated_features[2,9]
   + 2.91114e-07 Conv_2_aggregated_features[2,10]
   + 9.46222e-06 Conv_2_aggregated_features[2,11]
   - 1.05406e-08 Conv_2_aggregated_features[2,12]
   - 0.3912120759487152 Conv_2_aggregated_features[2,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[2,14]
   - 9.82623e-10 Conv_2_aggregated_features[2,15] + Conv_2_t[2,14]
   = 0.3463285267353058
 Conv_2_output_constraint[2,15]: 2.91582e-06 Conv_1_Relu_ts[2,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[2,2]
   - 6.96981e-13 Conv_1_Relu_ts[2,5] + 1.06195e-10 Conv_1_Relu_ts[2,6]
   + 9.12088e-06 Conv_1_Relu_ts[2,9] - 7.00879e-07 Conv_1_Relu_ts[2,10]
   + 2.54492e-08 Conv_1_Relu_ts[2,11] - 8.8874e-07 Conv_1_Relu_ts[2,12]
   - 9.8182e-06 Conv_1_Relu_ts[2,13] + 1.53315e-07 Conv_1_Relu_ts[2,14]
   - 2.41261e-08 Conv_1_Relu_ts[2,15]
   + 3.09046e-06 Conv_2_aggregated_features[2,1]
   - 5.23654e-07 Conv_2_aggregated_features[2,2]
   + 3.40881e-13 Conv_2_aggregated_features[2,5]
   + 4.84763e-12 Conv_2_aggregated_features[2,6]
   + 6.94651e-06 Conv_2_aggregated_features[2,9]
   + 1.80346e-08 Conv_2_aggregated_features[2,10]
   - 8.89515e-09 Conv_2_aggregated_features[2,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[2,12]
   - 5.75117e-06 Conv_2_aggregated_features[2,13]
   - 3.11447e-07 Conv_2_aggregated_features[2,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[2,15]
   + Conv_2_t[2,15] = -5.06887e-06
 Conv_2_output_constraint[3,0]: 5.62815e-10 Conv_1_Relu_ts[3,7]
   + 1.92775e-11 Conv_1_Relu_ts[3,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[3,15]
   + 5.32965e-11 Conv_2_aggregated_features[3,7]
   - 3.50224e-08 Conv_2_aggregated_features[3,15] + Conv_2_t[3,0]
   = 2.0494e-14
 Conv_2_output_constraint[3,1]: - 3.1209e-05 Conv_1_Relu_ts[3,1]
   + 3.2097e-12 Conv_1_Relu_ts[3,2] + 6.75828e-11 Conv_1_Relu_ts[3,4]
   + 6.05596e-11 Conv_1_Relu_ts[3,5] + 9.57194e-07 Conv_1_Relu_ts[3,6]
   - 1.94554e-13 Conv_1_Relu_ts[3,7] + 9.60207e-05 Conv_1_Relu_ts[3,9]
   + 3.54229e-11 Conv_1_Relu_ts[3,10] + 2.0034e-09 Conv_1_Relu_ts[3,11]
   + 3.41256e-05 Conv_1_Relu_ts[3,13] + 2.24541e-06 Conv_1_Relu_ts[3,14]
   - 7.56678e-10 Conv_1_Relu_ts[3,15]
   - 3.63258e-06 Conv_2_aggregated_features[3,1]
   - 2.02372e-12 Conv_2_aggregated_features[3,2]
   - 6.7717e-13 Conv_2_aggregated_features[3,4]
   + 1.64055e-10 Conv_2_aggregated_features[3,5]
   + 8.63851e-05 Conv_2_aggregated_features[3,6]
   - 5.43338e-10 Conv_2_aggregated_features[3,7]
   + 8.87988e-05 Conv_2_aggregated_features[3,9]
   + 2.03334e-11 Conv_2_aggregated_features[3,10]
   + 2.09865e-09 Conv_2_aggregated_features[3,11]
   + 5.1289e-05 Conv_2_aggregated_features[3,13]
   - 1.99215e-06 Conv_2_aggregated_features[3,14]
   - 2.85245e-06 Conv_2_aggregated_features[3,15] + Conv_2_t[3,1]
   = -5.09804e-05
 Conv_2_output_constraint[3,2]: 4.6554e-12 Conv_1_Relu_ts[3,1]
   - 5.08004e-07 Conv_1_Relu_ts[3,4] + 8.2088e-12 Conv_1_Relu_ts[3,9]
   + 3.89405e-05 Conv_1_Relu_ts[3,10] + 3.83811e-12 Conv_1_Relu_ts[3,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[3,15]
   + 2.16136e-12 Conv_2_aggregated_features[3,1]
   - 4.18879e-13 Conv_2_aggregated_features[3,4]
   + 1.01588e-11 Conv_2_aggregated_features[3,9]
   + 4.38408e-12 Conv_2_aggregated_features[3,13]
   + 6.90517e-06 Conv_2_aggregated_features[3,15] + Conv_2_t[3,2]
   = -1.21231e-11
 Conv_2_output_constraint[3,3]: - 9.77857e-07 Conv_1_Relu_ts[3,1]
   - 1.62341e-06 Conv_1_Relu_ts[3,4] - 1.26004e-09 Conv_1_Relu_ts[3,6]
   + 2.40003e-07 Conv_1_Relu_ts[3,7] - 5.41958e-07 Conv_1_Relu_ts[3,9]
   + 1.16363e-09 Conv_1_Relu_ts[3,10] - 2.13281e-11 Conv_1_Relu_ts[3,11]
   + 4.12955e-09 Conv_1_Relu_ts[3,12] - 3.71195e-07 Conv_1_Relu_ts[3,13]
   + 2.2241e-11 Conv_1_Relu_ts[3,14] + 6.37491e-09 Conv_1_Relu_ts[3,15]
   - 1.2731e-06 Conv_2_aggregated_features[3,1]
   - 1.0258e-06 Conv_2_aggregated_features[3,4]
   + 4.06095e-10 Conv_2_aggregated_features[3,6]
   + 3.12011e-07 Conv_2_aggregated_features[3,7]
   - 3.09535e-07 Conv_2_aggregated_features[3,9]
   - 0.00177046051248908 Conv_2_aggregated_features[3,10]
   + 8.54383e-13 Conv_2_aggregated_features[3,11]
   - 3.03582e-06 Conv_2_aggregated_features[3,12]
   - 4.41436e-07 Conv_2_aggregated_features[3,13]
   + 7.0584e-12 Conv_2_aggregated_features[3,14]
   - 9.96128e-05 Conv_2_aggregated_features[3,15] + Conv_2_t[3,3]
   = 4.02827e-07
 Conv_2_output_constraint[3,4]: 5.03273e-07 Conv_1_Relu_ts[3,1]
   - 6.22027e-09 Conv_1_Relu_ts[3,5] - 1.47914e-11 Conv_1_Relu_ts[3,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[3,9]
   + 1.18025e-13 Conv_1_Relu_ts[3,10] + 9.23352e-05 Conv_1_Relu_ts[3,11]
   + 0.000518991 Conv_1_Relu_ts[3,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[3,14]
   - 1.05921e-06 Conv_2_aggregated_features[3,1]
   + 2.88866e-10 Conv_2_aggregated_features[3,5]
   - 2.58968e-11 Conv_2_aggregated_features[3,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[3,9]
   - 3.93262e-05 Conv_2_aggregated_features[3,11]
   + 0.000690713 Conv_2_aggregated_features[3,13]
   + 7.70641e-05 Conv_2_aggregated_features[3,14] + Conv_2_t[3,4]
   = -5.84568e-07
 Conv_2_output_constraint[3,5]: 4.37405e-10 Conv_1_Relu_ts[3,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[3,4]
   + 1.77244e-09 Conv_1_Relu_ts[3,5] + 6.99595e-10 Conv_1_Relu_ts[3,9]
   - 1.77001e-13 Conv_1_Relu_ts[3,11] + 2.86208e-12 Conv_1_Relu_ts[3,12]
   + 3.24683e-10 Conv_1_Relu_ts[3,13] - 6.56311e-06 Conv_1_Relu_ts[3,15]
   - 4.63786e-09 Conv_2_aggregated_features[3,0]
   - 4.27797e-10 Conv_2_aggregated_features[3,1]
   - 2.50277e-13 Conv_2_aggregated_features[3,4]
   - 2.35888e-05 Conv_2_aggregated_features[3,5]
   + 6.12624e-10 Conv_2_aggregated_features[3,9]
   + 3.98546e-10 Conv_2_aggregated_features[3,13]
   + 9.89818e-10 Conv_2_aggregated_features[3,15] + Conv_2_t[3,5]
   = 6.24238e-10
 Conv_2_output_constraint[3,6]: - 4.80371e-09 Conv_1_Relu_ts[3,0]
   + 0.2580605149269104 Conv_1_Relu_ts[3,1]
   + 6.84534e-09 Conv_1_Relu_ts[3,2] + 7.44651e-09 Conv_1_Relu_ts[3,3]
   - 2.26729e-08 Conv_1_Relu_ts[3,4] + 1.08868e-09 Conv_1_Relu_ts[3,5]
   + 0.000151346 Conv_1_Relu_ts[3,6] - 1.07095e-09 Conv_1_Relu_ts[3,7]
   - 0.4001754522323608 Conv_1_Relu_ts[3,9]
   - 6.82816e-07 Conv_1_Relu_ts[3,10] - 3.7591e-05 Conv_1_Relu_ts[3,11]
   - 1.62843e-08 Conv_1_Relu_ts[3,12]
   + 1.447632431983948 Conv_1_Relu_ts[3,13]
   - 1.73427e-05 Conv_1_Relu_ts[3,14] + 2.28396e-09 Conv_1_Relu_ts[3,15]
   - 5.33024e-08 Conv_2_aggregated_features[3,0]
   + 0.8234332203865051 Conv_2_aggregated_features[3,1]
   + 6.68388e-09 Conv_2_aggregated_features[3,2]
   - 4.97866e-07 Conv_2_aggregated_features[3,3]
   - 2.61287e-08 Conv_2_aggregated_features[3,4]
   + 5.03108e-11 Conv_2_aggregated_features[3,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[3,6]
   - 1.69808e-09 Conv_2_aggregated_features[3,7]
   - 1.210363864898682 Conv_2_aggregated_features[3,9]
   - 7.75092e-07 Conv_2_aggregated_features[3,10]
   - 6.16735e-05 Conv_2_aggregated_features[3,11]
   - 1.9235e-08 Conv_2_aggregated_features[3,12]
   + 0.4373943507671356 Conv_2_aggregated_features[3,13]
   - 0.00871682073920965 Conv_2_aggregated_features[3,14]
   + 1.60146e-09 Conv_2_aggregated_features[3,15] + Conv_2_t[3,6]
   = 0.1640831530094147
 Conv_2_output_constraint[3,7]: 0.1483074277639389 Conv_1_Relu_ts[3,1]
   + 1.25136e-09 Conv_1_Relu_ts[3,2] - 3.02443e-09 Conv_1_Relu_ts[3,4]
   + 1.45591e-10 Conv_1_Relu_ts[3,5] - 3.6559e-05 Conv_1_Relu_ts[3,6]
   - 7.95116e-08 Conv_1_Relu_ts[3,7]
   + 0.0743252262473106 Conv_1_Relu_ts[3,9]
   + 2.19629e-08 Conv_1_Relu_ts[3,10] - 1.95369e-05 Conv_1_Relu_ts[3,11]
   - 1.21901e-08 Conv_1_Relu_ts[3,12]
   - 0.3415087759494781 Conv_1_Relu_ts[3,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[3,14]
   - 3.54178e-10 Conv_1_Relu_ts[3,15]
   + 0.2313222438097 Conv_2_aggregated_features[3,1]
   + 1.07351e-09 Conv_2_aggregated_features[3,2]
   - 4.05873e-09 Conv_2_aggregated_features[3,4]
   + 1.25187e-10 Conv_2_aggregated_features[3,5]
   - 3.00173e-05 Conv_2_aggregated_features[3,6]
   - 6.57874e-08 Conv_2_aggregated_features[3,7]
   + 0.0370698347687721 Conv_2_aggregated_features[3,9]
   - 2.2615e-08 Conv_2_aggregated_features[3,10]
   - 2.0281e-05 Conv_2_aggregated_features[3,11]
   - 1.14356e-08 Conv_2_aggregated_features[3,12]
   - 0.7083228826522827 Conv_2_aggregated_features[3,13]
   + 0.00042985 Conv_2_aggregated_features[3,14]
   - 4.8015e-10 Conv_2_aggregated_features[3,15] + Conv_2_t[3,7]
   = -0.5274350643157959
 Conv_2_output_constraint[3,8]: - 3.50628e-08 Conv_1_Relu_ts[3,1]
   - 0.00127368199173361 Conv_1_Relu_ts[3,5]
   + 1.1216e-10 Conv_1_Relu_ts[3,6] + 9.94055e-06 Conv_1_Relu_ts[3,9]
   + 2.4753e-05 Conv_1_Relu_ts[3,11] + 2.36679e-06 Conv_1_Relu_ts[3,13]
   + 1.90767e-09 Conv_1_Relu_ts[3,14] - 1.56108e-07 Conv_1_Relu_ts[3,15]
   - 5.18578e-06 Conv_2_aggregated_features[3,1]
   - 2.22254e-05 Conv_2_aggregated_features[3,5]
   + 9.48265e-11 Conv_2_aggregated_features[3,6]
   + 3.14595e-06 Conv_2_aggregated_features[3,9]
   - 1.16025e-05 Conv_2_aggregated_features[3,11]
   + 5.35568e-06 Conv_2_aggregated_features[3,13]
   + 1.47492e-09 Conv_2_aggregated_features[3,14]
   + 1.78245e-06 Conv_2_aggregated_features[3,15] + Conv_2_t[3,8]
   = 1.33952e-09
 Conv_2_output_constraint[3,9]: - 3.05926e-05 Conv_1_Relu_ts[3,1]
   - 4.69457e-10 Conv_1_Relu_ts[3,3] - 1.88994e-07 Conv_1_Relu_ts[3,5]
   + 1.14687e-08 Conv_1_Relu_ts[3,6] - 2.26795e-08 Conv_1_Relu_ts[3,7]
   - 1.1875e-05 Conv_1_Relu_ts[3,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[3,10]
   + 1.3279e-05 Conv_1_Relu_ts[3,11] - 8.03594e-13 Conv_1_Relu_ts[3,12]
   - 1.23268e-05 Conv_1_Relu_ts[3,13] - 1.4779e-07 Conv_1_Relu_ts[3,14]
   - 3.81296e-05 Conv_1_Relu_ts[3,15]
   + 3.05076e-11 Conv_2_aggregated_features[3,0]
   - 1.09971e-05 Conv_2_aggregated_features[3,1]
   - 6.1268e-09 Conv_2_aggregated_features[3,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[3,5]
   + 3.70939e-09 Conv_2_aggregated_features[3,6]
   - 2.02588e-06 Conv_2_aggregated_features[3,7]
   - 1.22094e-05 Conv_2_aggregated_features[3,9]
   - 1.30108e-06 Conv_2_aggregated_features[3,10]
   + 1.18736e-05 Conv_2_aggregated_features[3,11]
   + 9.38886e-13 Conv_2_aggregated_features[3,12]
   - 1.31367e-05 Conv_2_aggregated_features[3,13]
   + 1.5916e-09 Conv_2_aggregated_features[3,14]
   + 5.00944e-10 Conv_2_aggregated_features[3,15] + Conv_2_t[3,9]
   = 2.10696e-05
 Conv_2_output_constraint[3,10]: 2.28334e-09 Conv_1_Relu_ts[3,1]
   - 1.39061e-09 Conv_1_Relu_ts[3,6] + 1.19969e-08 Conv_1_Relu_ts[3,9]
   - 5.35695e-08 Conv_1_Relu_ts[3,10] + 7.265e-09 Conv_1_Relu_ts[3,13]
   - 2.17292e-11 Conv_1_Relu_ts[3,14] + 1.05271e-10 Conv_1_Relu_ts[3,15]
   + 4.90141e-09 Conv_2_aggregated_features[3,1]
   + 6.88e-06 Conv_2_aggregated_features[3,6]
   + 6.7107e-09 Conv_2_aggregated_features[3,9]
   + 1.08045e-08 Conv_2_aggregated_features[3,13]
   - 1.00381e-11 Conv_2_aggregated_features[3,14]
   - 5.27939e-06 Conv_2_aggregated_features[3,15] + Conv_2_t[3,10]
   = -6.36506e-09
 Conv_2_output_constraint[3,11]: - 9.57817e-09 Conv_1_Relu_ts[3,1]
   + 2.02195e-07 Conv_1_Relu_ts[3,2] - 3.79923e-06 Conv_1_Relu_ts[3,5]
   + 5.56008e-05 Conv_1_Relu_ts[3,6] - 5.0366e-09 Conv_1_Relu_ts[3,9]
   - 7.52285e-09 Conv_1_Relu_ts[3,13] - 5.6608e-10 Conv_1_Relu_ts[3,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[3,15]
   - 7.76309e-09 Conv_2_aggregated_features[3,1]
   - 2.62087e-08 Conv_2_aggregated_features[3,2]
   - 3.59639e-06 Conv_2_aggregated_features[3,5]
   - 2.75699e-10 Conv_2_aggregated_features[3,6]
   - 5.01117e-09 Conv_2_aggregated_features[3,9]
   - 7.1903e-09 Conv_2_aggregated_features[3,13]
   - 1.46623e-10 Conv_2_aggregated_features[3,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[3,15]
   + Conv_2_t[3,11] = 3.19541e-09
 Conv_2_output_constraint[3,12]: 1.20473e-05 Conv_1_Relu_ts[3,0]
   + 0.240223228931427 Conv_1_Relu_ts[3,1] + 1.328e-09 Conv_1_Relu_ts[3,2]
   + 2.75965e-08 Conv_1_Relu_ts[3,4] + 1.3558e-09 Conv_1_Relu_ts[3,5]
   - 1.34848e-06 Conv_1_Relu_ts[3,6] + 4.96526e-10 Conv_1_Relu_ts[3,7]
   - 0.3355287313461304 Conv_1_Relu_ts[3,9]
   + 2.31291e-07 Conv_1_Relu_ts[3,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[3,11]
   + 1.69037e-09 Conv_1_Relu_ts[3,12]
   - 0.1011878773570061 Conv_1_Relu_ts[3,13]
   - 0.0104798665270209 Conv_1_Relu_ts[3,14]
   - 4.11824e-11 Conv_1_Relu_ts[3,15]
   + 1.48158e-05 Conv_2_aggregated_features[3,0]
   - 1.069679498672485 Conv_2_aggregated_features[3,1]
   + 1.35995e-09 Conv_2_aggregated_features[3,2]
   + 2.54971e-08 Conv_2_aggregated_features[3,4]
   + 1.4479e-09 Conv_2_aggregated_features[3,5]
   - 2.57874e-05 Conv_2_aggregated_features[3,6]
   + 5.82765e-10 Conv_2_aggregated_features[3,7]
   + 0.2310869246721268 Conv_2_aggregated_features[3,9]
   + 2.42688e-07 Conv_2_aggregated_features[3,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[3,11]
   + 1.80254e-09 Conv_2_aggregated_features[3,12]
   + 0.1041423305869102 Conv_2_aggregated_features[3,13]
   - 0.00980580504983664 Conv_2_aggregated_features[3,14]
   + 4.55897e-11 Conv_2_aggregated_features[3,15] + Conv_2_t[3,12]
   = -0.0243597943335772
 Conv_2_output_constraint[3,13]: - 6.73028e-05 Conv_1_Relu_ts[3,1]
   + 6.15243e-11 Conv_1_Relu_ts[3,2] + 7.12143e-10 Conv_1_Relu_ts[3,4]
   - 3.05256e-08 Conv_1_Relu_ts[3,6]
   + 0.0179135110229254 Conv_1_Relu_ts[3,9]
   - 1.78315e-08 Conv_1_Relu_ts[3,10] + 3.95661e-08 Conv_1_Relu_ts[3,11]
   + 4.14048e-11 Conv_1_Relu_ts[3,12]
   - 0.00573352118954062 Conv_1_Relu_ts[3,13]
   - 1.5325e-05 Conv_1_Relu_ts[3,14]
   - 0.0551034025847912 Conv_2_aggregated_features[3,1]
   + 6.93423e-11 Conv_2_aggregated_features[3,2]
   + 2.18846e-10 Conv_2_aggregated_features[3,4]
   + 1.23153e-11 Conv_2_aggregated_features[3,5]
   - 9.5689e-08 Conv_2_aggregated_features[3,6]
   + 1.83226e-11 Conv_2_aggregated_features[3,7]
   + 0.0415055826306343 Conv_2_aggregated_features[3,9]
   - 4.74355e-09 Conv_2_aggregated_features[3,10]
   - 2.92725e-08 Conv_2_aggregated_features[3,11]
   + 3.37044e-11 Conv_2_aggregated_features[3,12]
   + 0.0228109527379274 Conv_2_aggregated_features[3,13]
   - 3.29092e-05 Conv_2_aggregated_features[3,14]
   - 2.71861e-12 Conv_2_aggregated_features[3,15] + Conv_2_t[3,13]
   = -0.0283597465604544
 Conv_2_output_constraint[3,14]: - 0.3400861024856567 Conv_1_Relu_ts[3,1]
   - 2.59888e-09 Conv_1_Relu_ts[3,2] + 1.10746e-09 Conv_1_Relu_ts[3,4]
   - 4.05832e-10 Conv_1_Relu_ts[3,5] - 2.28438e-05 Conv_1_Relu_ts[3,6]
   - 7.66288e-10 Conv_1_Relu_ts[3,7]
   - 0.1812356412410736 Conv_1_Relu_ts[3,9]
   + 2.36737e-07 Conv_1_Relu_ts[3,10] + 5.46304e-06 Conv_1_Relu_ts[3,11]
   - 1.05702e-08 Conv_1_Relu_ts[3,12]
   - 0.1597411930561066 Conv_1_Relu_ts[3,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[3,14]
   - 1.23647e-09 Conv_1_Relu_ts[3,15]
   - 0.0805128961801529 Conv_2_aggregated_features[3,1]
   - 2.67203e-09 Conv_2_aggregated_features[3,2]
   + 4.58926e-09 Conv_2_aggregated_features[3,4]
   - 5.32055e-10 Conv_2_aggregated_features[3,5]
   - 2.77383e-05 Conv_2_aggregated_features[3,6]
   - 5.76852e-10 Conv_2_aggregated_features[3,7]
   - 0.301887035369873 Conv_2_aggregated_features[3,9]
   + 2.91114e-07 Conv_2_aggregated_features[3,10]
   + 9.46222e-06 Conv_2_aggregated_features[3,11]
   - 1.05406e-08 Conv_2_aggregated_features[3,12]
   - 0.3912120759487152 Conv_2_aggregated_features[3,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[3,14]
   - 9.82623e-10 Conv_2_aggregated_features[3,15] + Conv_2_t[3,14]
   = 0.3463285267353058
 Conv_2_output_constraint[3,15]: 2.91582e-06 Conv_1_Relu_ts[3,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[3,2]
   - 6.96981e-13 Conv_1_Relu_ts[3,5] + 1.06195e-10 Conv_1_Relu_ts[3,6]
   + 9.12088e-06 Conv_1_Relu_ts[3,9] - 7.00879e-07 Conv_1_Relu_ts[3,10]
   + 2.54492e-08 Conv_1_Relu_ts[3,11] - 8.8874e-07 Conv_1_Relu_ts[3,12]
   - 9.8182e-06 Conv_1_Relu_ts[3,13] + 1.53315e-07 Conv_1_Relu_ts[3,14]
   - 2.41261e-08 Conv_1_Relu_ts[3,15]
   + 3.09046e-06 Conv_2_aggregated_features[3,1]
   - 5.23654e-07 Conv_2_aggregated_features[3,2]
   + 3.40881e-13 Conv_2_aggregated_features[3,5]
   + 4.84763e-12 Conv_2_aggregated_features[3,6]
   + 6.94651e-06 Conv_2_aggregated_features[3,9]
   + 1.80346e-08 Conv_2_aggregated_features[3,10]
   - 8.89515e-09 Conv_2_aggregated_features[3,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[3,12]
   - 5.75117e-06 Conv_2_aggregated_features[3,13]
   - 3.11447e-07 Conv_2_aggregated_features[3,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[3,15]
   + Conv_2_t[3,15] = -5.06887e-06
 Conv_2_output_constraint[4,0]: 5.62815e-10 Conv_1_Relu_ts[4,7]
   + 1.92775e-11 Conv_1_Relu_ts[4,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[4,15]
   + 5.32965e-11 Conv_2_aggregated_features[4,7]
   - 3.50224e-08 Conv_2_aggregated_features[4,15] + Conv_2_t[4,0]
   = 2.0494e-14
 Conv_2_output_constraint[4,1]: - 3.1209e-05 Conv_1_Relu_ts[4,1]
   + 3.2097e-12 Conv_1_Relu_ts[4,2] + 6.75828e-11 Conv_1_Relu_ts[4,4]
   + 6.05596e-11 Conv_1_Relu_ts[4,5] + 9.57194e-07 Conv_1_Relu_ts[4,6]
   - 1.94554e-13 Conv_1_Relu_ts[4,7] + 9.60207e-05 Conv_1_Relu_ts[4,9]
   + 3.54229e-11 Conv_1_Relu_ts[4,10] + 2.0034e-09 Conv_1_Relu_ts[4,11]
   + 3.41256e-05 Conv_1_Relu_ts[4,13] + 2.24541e-06 Conv_1_Relu_ts[4,14]
   - 7.56678e-10 Conv_1_Relu_ts[4,15]
   - 3.63258e-06 Conv_2_aggregated_features[4,1]
   - 2.02372e-12 Conv_2_aggregated_features[4,2]
   - 6.7717e-13 Conv_2_aggregated_features[4,4]
   + 1.64055e-10 Conv_2_aggregated_features[4,5]
   + 8.63851e-05 Conv_2_aggregated_features[4,6]
   - 5.43338e-10 Conv_2_aggregated_features[4,7]
   + 8.87988e-05 Conv_2_aggregated_features[4,9]
   + 2.03334e-11 Conv_2_aggregated_features[4,10]
   + 2.09865e-09 Conv_2_aggregated_features[4,11]
   + 5.1289e-05 Conv_2_aggregated_features[4,13]
   - 1.99215e-06 Conv_2_aggregated_features[4,14]
   - 2.85245e-06 Conv_2_aggregated_features[4,15] + Conv_2_t[4,1]
   = -5.09804e-05
 Conv_2_output_constraint[4,2]: 4.6554e-12 Conv_1_Relu_ts[4,1]
   - 5.08004e-07 Conv_1_Relu_ts[4,4] + 8.2088e-12 Conv_1_Relu_ts[4,9]
   + 3.89405e-05 Conv_1_Relu_ts[4,10] + 3.83811e-12 Conv_1_Relu_ts[4,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[4,15]
   + 2.16136e-12 Conv_2_aggregated_features[4,1]
   - 4.18879e-13 Conv_2_aggregated_features[4,4]
   + 1.01588e-11 Conv_2_aggregated_features[4,9]
   + 4.38408e-12 Conv_2_aggregated_features[4,13]
   + 6.90517e-06 Conv_2_aggregated_features[4,15] + Conv_2_t[4,2]
   = -1.21231e-11
 Conv_2_output_constraint[4,3]: - 9.77857e-07 Conv_1_Relu_ts[4,1]
   - 1.62341e-06 Conv_1_Relu_ts[4,4] - 1.26004e-09 Conv_1_Relu_ts[4,6]
   + 2.40003e-07 Conv_1_Relu_ts[4,7] - 5.41958e-07 Conv_1_Relu_ts[4,9]
   + 1.16363e-09 Conv_1_Relu_ts[4,10] - 2.13281e-11 Conv_1_Relu_ts[4,11]
   + 4.12955e-09 Conv_1_Relu_ts[4,12] - 3.71195e-07 Conv_1_Relu_ts[4,13]
   + 2.2241e-11 Conv_1_Relu_ts[4,14] + 6.37491e-09 Conv_1_Relu_ts[4,15]
   - 1.2731e-06 Conv_2_aggregated_features[4,1]
   - 1.0258e-06 Conv_2_aggregated_features[4,4]
   + 4.06095e-10 Conv_2_aggregated_features[4,6]
   + 3.12011e-07 Conv_2_aggregated_features[4,7]
   - 3.09535e-07 Conv_2_aggregated_features[4,9]
   - 0.00177046051248908 Conv_2_aggregated_features[4,10]
   + 8.54383e-13 Conv_2_aggregated_features[4,11]
   - 3.03582e-06 Conv_2_aggregated_features[4,12]
   - 4.41436e-07 Conv_2_aggregated_features[4,13]
   + 7.0584e-12 Conv_2_aggregated_features[4,14]
   - 9.96128e-05 Conv_2_aggregated_features[4,15] + Conv_2_t[4,3]
   = 4.02827e-07
 Conv_2_output_constraint[4,4]: 5.03273e-07 Conv_1_Relu_ts[4,1]
   - 6.22027e-09 Conv_1_Relu_ts[4,5] - 1.47914e-11 Conv_1_Relu_ts[4,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[4,9]
   + 1.18025e-13 Conv_1_Relu_ts[4,10] + 9.23352e-05 Conv_1_Relu_ts[4,11]
   + 0.000518991 Conv_1_Relu_ts[4,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[4,14]
   - 1.05921e-06 Conv_2_aggregated_features[4,1]
   + 2.88866e-10 Conv_2_aggregated_features[4,5]
   - 2.58968e-11 Conv_2_aggregated_features[4,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[4,9]
   - 3.93262e-05 Conv_2_aggregated_features[4,11]
   + 0.000690713 Conv_2_aggregated_features[4,13]
   + 7.70641e-05 Conv_2_aggregated_features[4,14] + Conv_2_t[4,4]
   = -5.84568e-07
 Conv_2_output_constraint[4,5]: 4.37405e-10 Conv_1_Relu_ts[4,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[4,4]
   + 1.77244e-09 Conv_1_Relu_ts[4,5] + 6.99595e-10 Conv_1_Relu_ts[4,9]
   - 1.77001e-13 Conv_1_Relu_ts[4,11] + 2.86208e-12 Conv_1_Relu_ts[4,12]
   + 3.24683e-10 Conv_1_Relu_ts[4,13] - 6.56311e-06 Conv_1_Relu_ts[4,15]
   - 4.63786e-09 Conv_2_aggregated_features[4,0]
   - 4.27797e-10 Conv_2_aggregated_features[4,1]
   - 2.50277e-13 Conv_2_aggregated_features[4,4]
   - 2.35888e-05 Conv_2_aggregated_features[4,5]
   + 6.12624e-10 Conv_2_aggregated_features[4,9]
   + 3.98546e-10 Conv_2_aggregated_features[4,13]
   + 9.89818e-10 Conv_2_aggregated_features[4,15] + Conv_2_t[4,5]
   = 6.24238e-10
 Conv_2_output_constraint[4,6]: - 4.80371e-09 Conv_1_Relu_ts[4,0]
   + 0.2580605149269104 Conv_1_Relu_ts[4,1]
   + 6.84534e-09 Conv_1_Relu_ts[4,2] + 7.44651e-09 Conv_1_Relu_ts[4,3]
   - 2.26729e-08 Conv_1_Relu_ts[4,4] + 1.08868e-09 Conv_1_Relu_ts[4,5]
   + 0.000151346 Conv_1_Relu_ts[4,6] - 1.07095e-09 Conv_1_Relu_ts[4,7]
   - 0.4001754522323608 Conv_1_Relu_ts[4,9]
   - 6.82816e-07 Conv_1_Relu_ts[4,10] - 3.7591e-05 Conv_1_Relu_ts[4,11]
   - 1.62843e-08 Conv_1_Relu_ts[4,12]
   + 1.447632431983948 Conv_1_Relu_ts[4,13]
   - 1.73427e-05 Conv_1_Relu_ts[4,14] + 2.28396e-09 Conv_1_Relu_ts[4,15]
   - 5.33024e-08 Conv_2_aggregated_features[4,0]
   + 0.8234332203865051 Conv_2_aggregated_features[4,1]
   + 6.68388e-09 Conv_2_aggregated_features[4,2]
   - 4.97866e-07 Conv_2_aggregated_features[4,3]
   - 2.61287e-08 Conv_2_aggregated_features[4,4]
   + 5.03108e-11 Conv_2_aggregated_features[4,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[4,6]
   - 1.69808e-09 Conv_2_aggregated_features[4,7]
   - 1.210363864898682 Conv_2_aggregated_features[4,9]
   - 7.75092e-07 Conv_2_aggregated_features[4,10]
   - 6.16735e-05 Conv_2_aggregated_features[4,11]
   - 1.9235e-08 Conv_2_aggregated_features[4,12]
   + 0.4373943507671356 Conv_2_aggregated_features[4,13]
   - 0.00871682073920965 Conv_2_aggregated_features[4,14]
   + 1.60146e-09 Conv_2_aggregated_features[4,15] + Conv_2_t[4,6]
   = 0.1640831530094147
 Conv_2_output_constraint[4,7]: 0.1483074277639389 Conv_1_Relu_ts[4,1]
   + 1.25136e-09 Conv_1_Relu_ts[4,2] - 3.02443e-09 Conv_1_Relu_ts[4,4]
   + 1.45591e-10 Conv_1_Relu_ts[4,5] - 3.6559e-05 Conv_1_Relu_ts[4,6]
   - 7.95116e-08 Conv_1_Relu_ts[4,7]
   + 0.0743252262473106 Conv_1_Relu_ts[4,9]
   + 2.19629e-08 Conv_1_Relu_ts[4,10] - 1.95369e-05 Conv_1_Relu_ts[4,11]
   - 1.21901e-08 Conv_1_Relu_ts[4,12]
   - 0.3415087759494781 Conv_1_Relu_ts[4,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[4,14]
   - 3.54178e-10 Conv_1_Relu_ts[4,15]
   + 0.2313222438097 Conv_2_aggregated_features[4,1]
   + 1.07351e-09 Conv_2_aggregated_features[4,2]
   - 4.05873e-09 Conv_2_aggregated_features[4,4]
   + 1.25187e-10 Conv_2_aggregated_features[4,5]
   - 3.00173e-05 Conv_2_aggregated_features[4,6]
   - 6.57874e-08 Conv_2_aggregated_features[4,7]
   + 0.0370698347687721 Conv_2_aggregated_features[4,9]
   - 2.2615e-08 Conv_2_aggregated_features[4,10]
   - 2.0281e-05 Conv_2_aggregated_features[4,11]
   - 1.14356e-08 Conv_2_aggregated_features[4,12]
   - 0.7083228826522827 Conv_2_aggregated_features[4,13]
   + 0.00042985 Conv_2_aggregated_features[4,14]
   - 4.8015e-10 Conv_2_aggregated_features[4,15] + Conv_2_t[4,7]
   = -0.5274350643157959
 Conv_2_output_constraint[4,8]: - 3.50628e-08 Conv_1_Relu_ts[4,1]
   - 0.00127368199173361 Conv_1_Relu_ts[4,5]
   + 1.1216e-10 Conv_1_Relu_ts[4,6] + 9.94055e-06 Conv_1_Relu_ts[4,9]
   + 2.4753e-05 Conv_1_Relu_ts[4,11] + 2.36679e-06 Conv_1_Relu_ts[4,13]
   + 1.90767e-09 Conv_1_Relu_ts[4,14] - 1.56108e-07 Conv_1_Relu_ts[4,15]
   - 5.18578e-06 Conv_2_aggregated_features[4,1]
   - 2.22254e-05 Conv_2_aggregated_features[4,5]
   + 9.48265e-11 Conv_2_aggregated_features[4,6]
   + 3.14595e-06 Conv_2_aggregated_features[4,9]
   - 1.16025e-05 Conv_2_aggregated_features[4,11]
   + 5.35568e-06 Conv_2_aggregated_features[4,13]
   + 1.47492e-09 Conv_2_aggregated_features[4,14]
   + 1.78245e-06 Conv_2_aggregated_features[4,15] + Conv_2_t[4,8]
   = 1.33952e-09
 Conv_2_output_constraint[4,9]: - 3.05926e-05 Conv_1_Relu_ts[4,1]
   - 4.69457e-10 Conv_1_Relu_ts[4,3] - 1.88994e-07 Conv_1_Relu_ts[4,5]
   + 1.14687e-08 Conv_1_Relu_ts[4,6] - 2.26795e-08 Conv_1_Relu_ts[4,7]
   - 1.1875e-05 Conv_1_Relu_ts[4,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[4,10]
   + 1.3279e-05 Conv_1_Relu_ts[4,11] - 8.03594e-13 Conv_1_Relu_ts[4,12]
   - 1.23268e-05 Conv_1_Relu_ts[4,13] - 1.4779e-07 Conv_1_Relu_ts[4,14]
   - 3.81296e-05 Conv_1_Relu_ts[4,15]
   + 3.05076e-11 Conv_2_aggregated_features[4,0]
   - 1.09971e-05 Conv_2_aggregated_features[4,1]
   - 6.1268e-09 Conv_2_aggregated_features[4,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[4,5]
   + 3.70939e-09 Conv_2_aggregated_features[4,6]
   - 2.02588e-06 Conv_2_aggregated_features[4,7]
   - 1.22094e-05 Conv_2_aggregated_features[4,9]
   - 1.30108e-06 Conv_2_aggregated_features[4,10]
   + 1.18736e-05 Conv_2_aggregated_features[4,11]
   + 9.38886e-13 Conv_2_aggregated_features[4,12]
   - 1.31367e-05 Conv_2_aggregated_features[4,13]
   + 1.5916e-09 Conv_2_aggregated_features[4,14]
   + 5.00944e-10 Conv_2_aggregated_features[4,15] + Conv_2_t[4,9]
   = 2.10696e-05
 Conv_2_output_constraint[4,10]: 2.28334e-09 Conv_1_Relu_ts[4,1]
   - 1.39061e-09 Conv_1_Relu_ts[4,6] + 1.19969e-08 Conv_1_Relu_ts[4,9]
   - 5.35695e-08 Conv_1_Relu_ts[4,10] + 7.265e-09 Conv_1_Relu_ts[4,13]
   - 2.17292e-11 Conv_1_Relu_ts[4,14] + 1.05271e-10 Conv_1_Relu_ts[4,15]
   + 4.90141e-09 Conv_2_aggregated_features[4,1]
   + 6.88e-06 Conv_2_aggregated_features[4,6]
   + 6.7107e-09 Conv_2_aggregated_features[4,9]
   + 1.08045e-08 Conv_2_aggregated_features[4,13]
   - 1.00381e-11 Conv_2_aggregated_features[4,14]
   - 5.27939e-06 Conv_2_aggregated_features[4,15] + Conv_2_t[4,10]
   = -6.36506e-09
 Conv_2_output_constraint[4,11]: - 9.57817e-09 Conv_1_Relu_ts[4,1]
   + 2.02195e-07 Conv_1_Relu_ts[4,2] - 3.79923e-06 Conv_1_Relu_ts[4,5]
   + 5.56008e-05 Conv_1_Relu_ts[4,6] - 5.0366e-09 Conv_1_Relu_ts[4,9]
   - 7.52285e-09 Conv_1_Relu_ts[4,13] - 5.6608e-10 Conv_1_Relu_ts[4,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[4,15]
   - 7.76309e-09 Conv_2_aggregated_features[4,1]
   - 2.62087e-08 Conv_2_aggregated_features[4,2]
   - 3.59639e-06 Conv_2_aggregated_features[4,5]
   - 2.75699e-10 Conv_2_aggregated_features[4,6]
   - 5.01117e-09 Conv_2_aggregated_features[4,9]
   - 7.1903e-09 Conv_2_aggregated_features[4,13]
   - 1.46623e-10 Conv_2_aggregated_features[4,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[4,15]
   + Conv_2_t[4,11] = 3.19541e-09
 Conv_2_output_constraint[4,12]: 1.20473e-05 Conv_1_Relu_ts[4,0]
   + 0.240223228931427 Conv_1_Relu_ts[4,1] + 1.328e-09 Conv_1_Relu_ts[4,2]
   + 2.75965e-08 Conv_1_Relu_ts[4,4] + 1.3558e-09 Conv_1_Relu_ts[4,5]
   - 1.34848e-06 Conv_1_Relu_ts[4,6] + 4.96526e-10 Conv_1_Relu_ts[4,7]
   - 0.3355287313461304 Conv_1_Relu_ts[4,9]
   + 2.31291e-07 Conv_1_Relu_ts[4,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[4,11]
   + 1.69037e-09 Conv_1_Relu_ts[4,12]
   - 0.1011878773570061 Conv_1_Relu_ts[4,13]
   - 0.0104798665270209 Conv_1_Relu_ts[4,14]
   - 4.11824e-11 Conv_1_Relu_ts[4,15]
   + 1.48158e-05 Conv_2_aggregated_features[4,0]
   - 1.069679498672485 Conv_2_aggregated_features[4,1]
   + 1.35995e-09 Conv_2_aggregated_features[4,2]
   + 2.54971e-08 Conv_2_aggregated_features[4,4]
   + 1.4479e-09 Conv_2_aggregated_features[4,5]
   - 2.57874e-05 Conv_2_aggregated_features[4,6]
   + 5.82765e-10 Conv_2_aggregated_features[4,7]
   + 0.2310869246721268 Conv_2_aggregated_features[4,9]
   + 2.42688e-07 Conv_2_aggregated_features[4,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[4,11]
   + 1.80254e-09 Conv_2_aggregated_features[4,12]
   + 0.1041423305869102 Conv_2_aggregated_features[4,13]
   - 0.00980580504983664 Conv_2_aggregated_features[4,14]
   + 4.55897e-11 Conv_2_aggregated_features[4,15] + Conv_2_t[4,12]
   = -0.0243597943335772
 Conv_2_output_constraint[4,13]: - 6.73028e-05 Conv_1_Relu_ts[4,1]
   + 6.15243e-11 Conv_1_Relu_ts[4,2] + 7.12143e-10 Conv_1_Relu_ts[4,4]
   - 3.05256e-08 Conv_1_Relu_ts[4,6]
   + 0.0179135110229254 Conv_1_Relu_ts[4,9]
   - 1.78315e-08 Conv_1_Relu_ts[4,10] + 3.95661e-08 Conv_1_Relu_ts[4,11]
   + 4.14048e-11 Conv_1_Relu_ts[4,12]
   - 0.00573352118954062 Conv_1_Relu_ts[4,13]
   - 1.5325e-05 Conv_1_Relu_ts[4,14]
   - 0.0551034025847912 Conv_2_aggregated_features[4,1]
   + 6.93423e-11 Conv_2_aggregated_features[4,2]
   + 2.18846e-10 Conv_2_aggregated_features[4,4]
   + 1.23153e-11 Conv_2_aggregated_features[4,5]
   - 9.5689e-08 Conv_2_aggregated_features[4,6]
   + 1.83226e-11 Conv_2_aggregated_features[4,7]
   + 0.0415055826306343 Conv_2_aggregated_features[4,9]
   - 4.74355e-09 Conv_2_aggregated_features[4,10]
   - 2.92725e-08 Conv_2_aggregated_features[4,11]
   + 3.37044e-11 Conv_2_aggregated_features[4,12]
   + 0.0228109527379274 Conv_2_aggregated_features[4,13]
   - 3.29092e-05 Conv_2_aggregated_features[4,14]
   - 2.71861e-12 Conv_2_aggregated_features[4,15] + Conv_2_t[4,13]
   = -0.0283597465604544
 Conv_2_output_constraint[4,14]: - 0.3400861024856567 Conv_1_Relu_ts[4,1]
   - 2.59888e-09 Conv_1_Relu_ts[4,2] + 1.10746e-09 Conv_1_Relu_ts[4,4]
   - 4.05832e-10 Conv_1_Relu_ts[4,5] - 2.28438e-05 Conv_1_Relu_ts[4,6]
   - 7.66288e-10 Conv_1_Relu_ts[4,7]
   - 0.1812356412410736 Conv_1_Relu_ts[4,9]
   + 2.36737e-07 Conv_1_Relu_ts[4,10] + 5.46304e-06 Conv_1_Relu_ts[4,11]
   - 1.05702e-08 Conv_1_Relu_ts[4,12]
   - 0.1597411930561066 Conv_1_Relu_ts[4,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[4,14]
   - 1.23647e-09 Conv_1_Relu_ts[4,15]
   - 0.0805128961801529 Conv_2_aggregated_features[4,1]
   - 2.67203e-09 Conv_2_aggregated_features[4,2]
   + 4.58926e-09 Conv_2_aggregated_features[4,4]
   - 5.32055e-10 Conv_2_aggregated_features[4,5]
   - 2.77383e-05 Conv_2_aggregated_features[4,6]
   - 5.76852e-10 Conv_2_aggregated_features[4,7]
   - 0.301887035369873 Conv_2_aggregated_features[4,9]
   + 2.91114e-07 Conv_2_aggregated_features[4,10]
   + 9.46222e-06 Conv_2_aggregated_features[4,11]
   - 1.05406e-08 Conv_2_aggregated_features[4,12]
   - 0.3912120759487152 Conv_2_aggregated_features[4,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[4,14]
   - 9.82623e-10 Conv_2_aggregated_features[4,15] + Conv_2_t[4,14]
   = 0.3463285267353058
 Conv_2_output_constraint[4,15]: 2.91582e-06 Conv_1_Relu_ts[4,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[4,2]
   - 6.96981e-13 Conv_1_Relu_ts[4,5] + 1.06195e-10 Conv_1_Relu_ts[4,6]
   + 9.12088e-06 Conv_1_Relu_ts[4,9] - 7.00879e-07 Conv_1_Relu_ts[4,10]
   + 2.54492e-08 Conv_1_Relu_ts[4,11] - 8.8874e-07 Conv_1_Relu_ts[4,12]
   - 9.8182e-06 Conv_1_Relu_ts[4,13] + 1.53315e-07 Conv_1_Relu_ts[4,14]
   - 2.41261e-08 Conv_1_Relu_ts[4,15]
   + 3.09046e-06 Conv_2_aggregated_features[4,1]
   - 5.23654e-07 Conv_2_aggregated_features[4,2]
   + 3.40881e-13 Conv_2_aggregated_features[4,5]
   + 4.84763e-12 Conv_2_aggregated_features[4,6]
   + 6.94651e-06 Conv_2_aggregated_features[4,9]
   + 1.80346e-08 Conv_2_aggregated_features[4,10]
   - 8.89515e-09 Conv_2_aggregated_features[4,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[4,12]
   - 5.75117e-06 Conv_2_aggregated_features[4,13]
   - 3.11447e-07 Conv_2_aggregated_features[4,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[4,15]
   + Conv_2_t[4,15] = -5.06887e-06
 Conv_2_output_constraint[5,0]: 5.62815e-10 Conv_1_Relu_ts[5,7]
   + 1.92775e-11 Conv_1_Relu_ts[5,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[5,15]
   + 5.32965e-11 Conv_2_aggregated_features[5,7]
   - 3.50224e-08 Conv_2_aggregated_features[5,15] + Conv_2_t[5,0]
   = 2.0494e-14
 Conv_2_output_constraint[5,1]: - 3.1209e-05 Conv_1_Relu_ts[5,1]
   + 3.2097e-12 Conv_1_Relu_ts[5,2] + 6.75828e-11 Conv_1_Relu_ts[5,4]
   + 6.05596e-11 Conv_1_Relu_ts[5,5] + 9.57194e-07 Conv_1_Relu_ts[5,6]
   - 1.94554e-13 Conv_1_Relu_ts[5,7] + 9.60207e-05 Conv_1_Relu_ts[5,9]
   + 3.54229e-11 Conv_1_Relu_ts[5,10] + 2.0034e-09 Conv_1_Relu_ts[5,11]
   + 3.41256e-05 Conv_1_Relu_ts[5,13] + 2.24541e-06 Conv_1_Relu_ts[5,14]
   - 7.56678e-10 Conv_1_Relu_ts[5,15]
   - 3.63258e-06 Conv_2_aggregated_features[5,1]
   - 2.02372e-12 Conv_2_aggregated_features[5,2]
   - 6.7717e-13 Conv_2_aggregated_features[5,4]
   + 1.64055e-10 Conv_2_aggregated_features[5,5]
   + 8.63851e-05 Conv_2_aggregated_features[5,6]
   - 5.43338e-10 Conv_2_aggregated_features[5,7]
   + 8.87988e-05 Conv_2_aggregated_features[5,9]
   + 2.03334e-11 Conv_2_aggregated_features[5,10]
   + 2.09865e-09 Conv_2_aggregated_features[5,11]
   + 5.1289e-05 Conv_2_aggregated_features[5,13]
   - 1.99215e-06 Conv_2_aggregated_features[5,14]
   - 2.85245e-06 Conv_2_aggregated_features[5,15] + Conv_2_t[5,1]
   = -5.09804e-05
 Conv_2_output_constraint[5,2]: 4.6554e-12 Conv_1_Relu_ts[5,1]
   - 5.08004e-07 Conv_1_Relu_ts[5,4] + 8.2088e-12 Conv_1_Relu_ts[5,9]
   + 3.89405e-05 Conv_1_Relu_ts[5,10] + 3.83811e-12 Conv_1_Relu_ts[5,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[5,15]
   + 2.16136e-12 Conv_2_aggregated_features[5,1]
   - 4.18879e-13 Conv_2_aggregated_features[5,4]
   + 1.01588e-11 Conv_2_aggregated_features[5,9]
   + 4.38408e-12 Conv_2_aggregated_features[5,13]
   + 6.90517e-06 Conv_2_aggregated_features[5,15] + Conv_2_t[5,2]
   = -1.21231e-11
 Conv_2_output_constraint[5,3]: - 9.77857e-07 Conv_1_Relu_ts[5,1]
   - 1.62341e-06 Conv_1_Relu_ts[5,4] - 1.26004e-09 Conv_1_Relu_ts[5,6]
   + 2.40003e-07 Conv_1_Relu_ts[5,7] - 5.41958e-07 Conv_1_Relu_ts[5,9]
   + 1.16363e-09 Conv_1_Relu_ts[5,10] - 2.13281e-11 Conv_1_Relu_ts[5,11]
   + 4.12955e-09 Conv_1_Relu_ts[5,12] - 3.71195e-07 Conv_1_Relu_ts[5,13]
   + 2.2241e-11 Conv_1_Relu_ts[5,14] + 6.37491e-09 Conv_1_Relu_ts[5,15]
   - 1.2731e-06 Conv_2_aggregated_features[5,1]
   - 1.0258e-06 Conv_2_aggregated_features[5,4]
   + 4.06095e-10 Conv_2_aggregated_features[5,6]
   + 3.12011e-07 Conv_2_aggregated_features[5,7]
   - 3.09535e-07 Conv_2_aggregated_features[5,9]
   - 0.00177046051248908 Conv_2_aggregated_features[5,10]
   + 8.54383e-13 Conv_2_aggregated_features[5,11]
   - 3.03582e-06 Conv_2_aggregated_features[5,12]
   - 4.41436e-07 Conv_2_aggregated_features[5,13]
   + 7.0584e-12 Conv_2_aggregated_features[5,14]
   - 9.96128e-05 Conv_2_aggregated_features[5,15] + Conv_2_t[5,3]
   = 4.02827e-07
 Conv_2_output_constraint[5,4]: 5.03273e-07 Conv_1_Relu_ts[5,1]
   - 6.22027e-09 Conv_1_Relu_ts[5,5] - 1.47914e-11 Conv_1_Relu_ts[5,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[5,9]
   + 1.18025e-13 Conv_1_Relu_ts[5,10] + 9.23352e-05 Conv_1_Relu_ts[5,11]
   + 0.000518991 Conv_1_Relu_ts[5,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[5,14]
   - 1.05921e-06 Conv_2_aggregated_features[5,1]
   + 2.88866e-10 Conv_2_aggregated_features[5,5]
   - 2.58968e-11 Conv_2_aggregated_features[5,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[5,9]
   - 3.93262e-05 Conv_2_aggregated_features[5,11]
   + 0.000690713 Conv_2_aggregated_features[5,13]
   + 7.70641e-05 Conv_2_aggregated_features[5,14] + Conv_2_t[5,4]
   = -5.84568e-07
 Conv_2_output_constraint[5,5]: 4.37405e-10 Conv_1_Relu_ts[5,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[5,4]
   + 1.77244e-09 Conv_1_Relu_ts[5,5] + 6.99595e-10 Conv_1_Relu_ts[5,9]
   - 1.77001e-13 Conv_1_Relu_ts[5,11] + 2.86208e-12 Conv_1_Relu_ts[5,12]
   + 3.24683e-10 Conv_1_Relu_ts[5,13] - 6.56311e-06 Conv_1_Relu_ts[5,15]
   - 4.63786e-09 Conv_2_aggregated_features[5,0]
   - 4.27797e-10 Conv_2_aggregated_features[5,1]
   - 2.50277e-13 Conv_2_aggregated_features[5,4]
   - 2.35888e-05 Conv_2_aggregated_features[5,5]
   + 6.12624e-10 Conv_2_aggregated_features[5,9]
   + 3.98546e-10 Conv_2_aggregated_features[5,13]
   + 9.89818e-10 Conv_2_aggregated_features[5,15] + Conv_2_t[5,5]
   = 6.24238e-10
 Conv_2_output_constraint[5,6]: - 4.80371e-09 Conv_1_Relu_ts[5,0]
   + 0.2580605149269104 Conv_1_Relu_ts[5,1]
   + 6.84534e-09 Conv_1_Relu_ts[5,2] + 7.44651e-09 Conv_1_Relu_ts[5,3]
   - 2.26729e-08 Conv_1_Relu_ts[5,4] + 1.08868e-09 Conv_1_Relu_ts[5,5]
   + 0.000151346 Conv_1_Relu_ts[5,6] - 1.07095e-09 Conv_1_Relu_ts[5,7]
   - 0.4001754522323608 Conv_1_Relu_ts[5,9]
   - 6.82816e-07 Conv_1_Relu_ts[5,10] - 3.7591e-05 Conv_1_Relu_ts[5,11]
   - 1.62843e-08 Conv_1_Relu_ts[5,12]
   + 1.447632431983948 Conv_1_Relu_ts[5,13]
   - 1.73427e-05 Conv_1_Relu_ts[5,14] + 2.28396e-09 Conv_1_Relu_ts[5,15]
   - 5.33024e-08 Conv_2_aggregated_features[5,0]
   + 0.8234332203865051 Conv_2_aggregated_features[5,1]
   + 6.68388e-09 Conv_2_aggregated_features[5,2]
   - 4.97866e-07 Conv_2_aggregated_features[5,3]
   - 2.61287e-08 Conv_2_aggregated_features[5,4]
   + 5.03108e-11 Conv_2_aggregated_features[5,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[5,6]
   - 1.69808e-09 Conv_2_aggregated_features[5,7]
   - 1.210363864898682 Conv_2_aggregated_features[5,9]
   - 7.75092e-07 Conv_2_aggregated_features[5,10]
   - 6.16735e-05 Conv_2_aggregated_features[5,11]
   - 1.9235e-08 Conv_2_aggregated_features[5,12]
   + 0.4373943507671356 Conv_2_aggregated_features[5,13]
   - 0.00871682073920965 Conv_2_aggregated_features[5,14]
   + 1.60146e-09 Conv_2_aggregated_features[5,15] + Conv_2_t[5,6]
   = 0.1640831530094147
 Conv_2_output_constraint[5,7]: 0.1483074277639389 Conv_1_Relu_ts[5,1]
   + 1.25136e-09 Conv_1_Relu_ts[5,2] - 3.02443e-09 Conv_1_Relu_ts[5,4]
   + 1.45591e-10 Conv_1_Relu_ts[5,5] - 3.6559e-05 Conv_1_Relu_ts[5,6]
   - 7.95116e-08 Conv_1_Relu_ts[5,7]
   + 0.0743252262473106 Conv_1_Relu_ts[5,9]
   + 2.19629e-08 Conv_1_Relu_ts[5,10] - 1.95369e-05 Conv_1_Relu_ts[5,11]
   - 1.21901e-08 Conv_1_Relu_ts[5,12]
   - 0.3415087759494781 Conv_1_Relu_ts[5,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[5,14]
   - 3.54178e-10 Conv_1_Relu_ts[5,15]
   + 0.2313222438097 Conv_2_aggregated_features[5,1]
   + 1.07351e-09 Conv_2_aggregated_features[5,2]
   - 4.05873e-09 Conv_2_aggregated_features[5,4]
   + 1.25187e-10 Conv_2_aggregated_features[5,5]
   - 3.00173e-05 Conv_2_aggregated_features[5,6]
   - 6.57874e-08 Conv_2_aggregated_features[5,7]
   + 0.0370698347687721 Conv_2_aggregated_features[5,9]
   - 2.2615e-08 Conv_2_aggregated_features[5,10]
   - 2.0281e-05 Conv_2_aggregated_features[5,11]
   - 1.14356e-08 Conv_2_aggregated_features[5,12]
   - 0.7083228826522827 Conv_2_aggregated_features[5,13]
   + 0.00042985 Conv_2_aggregated_features[5,14]
   - 4.8015e-10 Conv_2_aggregated_features[5,15] + Conv_2_t[5,7]
   = -0.5274350643157959
 Conv_2_output_constraint[5,8]: - 3.50628e-08 Conv_1_Relu_ts[5,1]
   - 0.00127368199173361 Conv_1_Relu_ts[5,5]
   + 1.1216e-10 Conv_1_Relu_ts[5,6] + 9.94055e-06 Conv_1_Relu_ts[5,9]
   + 2.4753e-05 Conv_1_Relu_ts[5,11] + 2.36679e-06 Conv_1_Relu_ts[5,13]
   + 1.90767e-09 Conv_1_Relu_ts[5,14] - 1.56108e-07 Conv_1_Relu_ts[5,15]
   - 5.18578e-06 Conv_2_aggregated_features[5,1]
   - 2.22254e-05 Conv_2_aggregated_features[5,5]
   + 9.48265e-11 Conv_2_aggregated_features[5,6]
   + 3.14595e-06 Conv_2_aggregated_features[5,9]
   - 1.16025e-05 Conv_2_aggregated_features[5,11]
   + 5.35568e-06 Conv_2_aggregated_features[5,13]
   + 1.47492e-09 Conv_2_aggregated_features[5,14]
   + 1.78245e-06 Conv_2_aggregated_features[5,15] + Conv_2_t[5,8]
   = 1.33952e-09
 Conv_2_output_constraint[5,9]: - 3.05926e-05 Conv_1_Relu_ts[5,1]
   - 4.69457e-10 Conv_1_Relu_ts[5,3] - 1.88994e-07 Conv_1_Relu_ts[5,5]
   + 1.14687e-08 Conv_1_Relu_ts[5,6] - 2.26795e-08 Conv_1_Relu_ts[5,7]
   - 1.1875e-05 Conv_1_Relu_ts[5,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[5,10]
   + 1.3279e-05 Conv_1_Relu_ts[5,11] - 8.03594e-13 Conv_1_Relu_ts[5,12]
   - 1.23268e-05 Conv_1_Relu_ts[5,13] - 1.4779e-07 Conv_1_Relu_ts[5,14]
   - 3.81296e-05 Conv_1_Relu_ts[5,15]
   + 3.05076e-11 Conv_2_aggregated_features[5,0]
   - 1.09971e-05 Conv_2_aggregated_features[5,1]
   - 6.1268e-09 Conv_2_aggregated_features[5,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[5,5]
   + 3.70939e-09 Conv_2_aggregated_features[5,6]
   - 2.02588e-06 Conv_2_aggregated_features[5,7]
   - 1.22094e-05 Conv_2_aggregated_features[5,9]
   - 1.30108e-06 Conv_2_aggregated_features[5,10]
   + 1.18736e-05 Conv_2_aggregated_features[5,11]
   + 9.38886e-13 Conv_2_aggregated_features[5,12]
   - 1.31367e-05 Conv_2_aggregated_features[5,13]
   + 1.5916e-09 Conv_2_aggregated_features[5,14]
   + 5.00944e-10 Conv_2_aggregated_features[5,15] + Conv_2_t[5,9]
   = 2.10696e-05
 Conv_2_output_constraint[5,10]: 2.28334e-09 Conv_1_Relu_ts[5,1]
   - 1.39061e-09 Conv_1_Relu_ts[5,6] + 1.19969e-08 Conv_1_Relu_ts[5,9]
   - 5.35695e-08 Conv_1_Relu_ts[5,10] + 7.265e-09 Conv_1_Relu_ts[5,13]
   - 2.17292e-11 Conv_1_Relu_ts[5,14] + 1.05271e-10 Conv_1_Relu_ts[5,15]
   + 4.90141e-09 Conv_2_aggregated_features[5,1]
   + 6.88e-06 Conv_2_aggregated_features[5,6]
   + 6.7107e-09 Conv_2_aggregated_features[5,9]
   + 1.08045e-08 Conv_2_aggregated_features[5,13]
   - 1.00381e-11 Conv_2_aggregated_features[5,14]
   - 5.27939e-06 Conv_2_aggregated_features[5,15] + Conv_2_t[5,10]
   = -6.36506e-09
 Conv_2_output_constraint[5,11]: - 9.57817e-09 Conv_1_Relu_ts[5,1]
   + 2.02195e-07 Conv_1_Relu_ts[5,2] - 3.79923e-06 Conv_1_Relu_ts[5,5]
   + 5.56008e-05 Conv_1_Relu_ts[5,6] - 5.0366e-09 Conv_1_Relu_ts[5,9]
   - 7.52285e-09 Conv_1_Relu_ts[5,13] - 5.6608e-10 Conv_1_Relu_ts[5,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[5,15]
   - 7.76309e-09 Conv_2_aggregated_features[5,1]
   - 2.62087e-08 Conv_2_aggregated_features[5,2]
   - 3.59639e-06 Conv_2_aggregated_features[5,5]
   - 2.75699e-10 Conv_2_aggregated_features[5,6]
   - 5.01117e-09 Conv_2_aggregated_features[5,9]
   - 7.1903e-09 Conv_2_aggregated_features[5,13]
   - 1.46623e-10 Conv_2_aggregated_features[5,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[5,15]
   + Conv_2_t[5,11] = 3.19541e-09
 Conv_2_output_constraint[5,12]: 1.20473e-05 Conv_1_Relu_ts[5,0]
   + 0.240223228931427 Conv_1_Relu_ts[5,1] + 1.328e-09 Conv_1_Relu_ts[5,2]
   + 2.75965e-08 Conv_1_Relu_ts[5,4] + 1.3558e-09 Conv_1_Relu_ts[5,5]
   - 1.34848e-06 Conv_1_Relu_ts[5,6] + 4.96526e-10 Conv_1_Relu_ts[5,7]
   - 0.3355287313461304 Conv_1_Relu_ts[5,9]
   + 2.31291e-07 Conv_1_Relu_ts[5,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[5,11]
   + 1.69037e-09 Conv_1_Relu_ts[5,12]
   - 0.1011878773570061 Conv_1_Relu_ts[5,13]
   - 0.0104798665270209 Conv_1_Relu_ts[5,14]
   - 4.11824e-11 Conv_1_Relu_ts[5,15]
   + 1.48158e-05 Conv_2_aggregated_features[5,0]
   - 1.069679498672485 Conv_2_aggregated_features[5,1]
   + 1.35995e-09 Conv_2_aggregated_features[5,2]
   + 2.54971e-08 Conv_2_aggregated_features[5,4]
   + 1.4479e-09 Conv_2_aggregated_features[5,5]
   - 2.57874e-05 Conv_2_aggregated_features[5,6]
   + 5.82765e-10 Conv_2_aggregated_features[5,7]
   + 0.2310869246721268 Conv_2_aggregated_features[5,9]
   + 2.42688e-07 Conv_2_aggregated_features[5,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[5,11]
   + 1.80254e-09 Conv_2_aggregated_features[5,12]
   + 0.1041423305869102 Conv_2_aggregated_features[5,13]
   - 0.00980580504983664 Conv_2_aggregated_features[5,14]
   + 4.55897e-11 Conv_2_aggregated_features[5,15] + Conv_2_t[5,12]
   = -0.0243597943335772
 Conv_2_output_constraint[5,13]: - 6.73028e-05 Conv_1_Relu_ts[5,1]
   + 6.15243e-11 Conv_1_Relu_ts[5,2] + 7.12143e-10 Conv_1_Relu_ts[5,4]
   - 3.05256e-08 Conv_1_Relu_ts[5,6]
   + 0.0179135110229254 Conv_1_Relu_ts[5,9]
   - 1.78315e-08 Conv_1_Relu_ts[5,10] + 3.95661e-08 Conv_1_Relu_ts[5,11]
   + 4.14048e-11 Conv_1_Relu_ts[5,12]
   - 0.00573352118954062 Conv_1_Relu_ts[5,13]
   - 1.5325e-05 Conv_1_Relu_ts[5,14]
   - 0.0551034025847912 Conv_2_aggregated_features[5,1]
   + 6.93423e-11 Conv_2_aggregated_features[5,2]
   + 2.18846e-10 Conv_2_aggregated_features[5,4]
   + 1.23153e-11 Conv_2_aggregated_features[5,5]
   - 9.5689e-08 Conv_2_aggregated_features[5,6]
   + 1.83226e-11 Conv_2_aggregated_features[5,7]
   + 0.0415055826306343 Conv_2_aggregated_features[5,9]
   - 4.74355e-09 Conv_2_aggregated_features[5,10]
   - 2.92725e-08 Conv_2_aggregated_features[5,11]
   + 3.37044e-11 Conv_2_aggregated_features[5,12]
   + 0.0228109527379274 Conv_2_aggregated_features[5,13]
   - 3.29092e-05 Conv_2_aggregated_features[5,14]
   - 2.71861e-12 Conv_2_aggregated_features[5,15] + Conv_2_t[5,13]
   = -0.0283597465604544
 Conv_2_output_constraint[5,14]: - 0.3400861024856567 Conv_1_Relu_ts[5,1]
   - 2.59888e-09 Conv_1_Relu_ts[5,2] + 1.10746e-09 Conv_1_Relu_ts[5,4]
   - 4.05832e-10 Conv_1_Relu_ts[5,5] - 2.28438e-05 Conv_1_Relu_ts[5,6]
   - 7.66288e-10 Conv_1_Relu_ts[5,7]
   - 0.1812356412410736 Conv_1_Relu_ts[5,9]
   + 2.36737e-07 Conv_1_Relu_ts[5,10] + 5.46304e-06 Conv_1_Relu_ts[5,11]
   - 1.05702e-08 Conv_1_Relu_ts[5,12]
   - 0.1597411930561066 Conv_1_Relu_ts[5,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[5,14]
   - 1.23647e-09 Conv_1_Relu_ts[5,15]
   - 0.0805128961801529 Conv_2_aggregated_features[5,1]
   - 2.67203e-09 Conv_2_aggregated_features[5,2]
   + 4.58926e-09 Conv_2_aggregated_features[5,4]
   - 5.32055e-10 Conv_2_aggregated_features[5,5]
   - 2.77383e-05 Conv_2_aggregated_features[5,6]
   - 5.76852e-10 Conv_2_aggregated_features[5,7]
   - 0.301887035369873 Conv_2_aggregated_features[5,9]
   + 2.91114e-07 Conv_2_aggregated_features[5,10]
   + 9.46222e-06 Conv_2_aggregated_features[5,11]
   - 1.05406e-08 Conv_2_aggregated_features[5,12]
   - 0.3912120759487152 Conv_2_aggregated_features[5,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[5,14]
   - 9.82623e-10 Conv_2_aggregated_features[5,15] + Conv_2_t[5,14]
   = 0.3463285267353058
 Conv_2_output_constraint[5,15]: 2.91582e-06 Conv_1_Relu_ts[5,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[5,2]
   - 6.96981e-13 Conv_1_Relu_ts[5,5] + 1.06195e-10 Conv_1_Relu_ts[5,6]
   + 9.12088e-06 Conv_1_Relu_ts[5,9] - 7.00879e-07 Conv_1_Relu_ts[5,10]
   + 2.54492e-08 Conv_1_Relu_ts[5,11] - 8.8874e-07 Conv_1_Relu_ts[5,12]
   - 9.8182e-06 Conv_1_Relu_ts[5,13] + 1.53315e-07 Conv_1_Relu_ts[5,14]
   - 2.41261e-08 Conv_1_Relu_ts[5,15]
   + 3.09046e-06 Conv_2_aggregated_features[5,1]
   - 5.23654e-07 Conv_2_aggregated_features[5,2]
   + 3.40881e-13 Conv_2_aggregated_features[5,5]
   + 4.84763e-12 Conv_2_aggregated_features[5,6]
   + 6.94651e-06 Conv_2_aggregated_features[5,9]
   + 1.80346e-08 Conv_2_aggregated_features[5,10]
   - 8.89515e-09 Conv_2_aggregated_features[5,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[5,12]
   - 5.75117e-06 Conv_2_aggregated_features[5,13]
   - 3.11447e-07 Conv_2_aggregated_features[5,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[5,15]
   + Conv_2_t[5,15] = -5.06887e-06
 Conv_2_output_constraint[6,0]: 5.62815e-10 Conv_1_Relu_ts[6,7]
   + 1.92775e-11 Conv_1_Relu_ts[6,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[6,15]
   + 5.32965e-11 Conv_2_aggregated_features[6,7]
   - 3.50224e-08 Conv_2_aggregated_features[6,15] + Conv_2_t[6,0]
   = 2.0494e-14
 Conv_2_output_constraint[6,1]: - 3.1209e-05 Conv_1_Relu_ts[6,1]
   + 3.2097e-12 Conv_1_Relu_ts[6,2] + 6.75828e-11 Conv_1_Relu_ts[6,4]
   + 6.05596e-11 Conv_1_Relu_ts[6,5] + 9.57194e-07 Conv_1_Relu_ts[6,6]
   - 1.94554e-13 Conv_1_Relu_ts[6,7] + 9.60207e-05 Conv_1_Relu_ts[6,9]
   + 3.54229e-11 Conv_1_Relu_ts[6,10] + 2.0034e-09 Conv_1_Relu_ts[6,11]
   + 3.41256e-05 Conv_1_Relu_ts[6,13] + 2.24541e-06 Conv_1_Relu_ts[6,14]
   - 7.56678e-10 Conv_1_Relu_ts[6,15]
   - 3.63258e-06 Conv_2_aggregated_features[6,1]
   - 2.02372e-12 Conv_2_aggregated_features[6,2]
   - 6.7717e-13 Conv_2_aggregated_features[6,4]
   + 1.64055e-10 Conv_2_aggregated_features[6,5]
   + 8.63851e-05 Conv_2_aggregated_features[6,6]
   - 5.43338e-10 Conv_2_aggregated_features[6,7]
   + 8.87988e-05 Conv_2_aggregated_features[6,9]
   + 2.03334e-11 Conv_2_aggregated_features[6,10]
   + 2.09865e-09 Conv_2_aggregated_features[6,11]
   + 5.1289e-05 Conv_2_aggregated_features[6,13]
   - 1.99215e-06 Conv_2_aggregated_features[6,14]
   - 2.85245e-06 Conv_2_aggregated_features[6,15] + Conv_2_t[6,1]
   = -5.09804e-05
 Conv_2_output_constraint[6,2]: 4.6554e-12 Conv_1_Relu_ts[6,1]
   - 5.08004e-07 Conv_1_Relu_ts[6,4] + 8.2088e-12 Conv_1_Relu_ts[6,9]
   + 3.89405e-05 Conv_1_Relu_ts[6,10] + 3.83811e-12 Conv_1_Relu_ts[6,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[6,15]
   + 2.16136e-12 Conv_2_aggregated_features[6,1]
   - 4.18879e-13 Conv_2_aggregated_features[6,4]
   + 1.01588e-11 Conv_2_aggregated_features[6,9]
   + 4.38408e-12 Conv_2_aggregated_features[6,13]
   + 6.90517e-06 Conv_2_aggregated_features[6,15] + Conv_2_t[6,2]
   = -1.21231e-11
 Conv_2_output_constraint[6,3]: - 9.77857e-07 Conv_1_Relu_ts[6,1]
   - 1.62341e-06 Conv_1_Relu_ts[6,4] - 1.26004e-09 Conv_1_Relu_ts[6,6]
   + 2.40003e-07 Conv_1_Relu_ts[6,7] - 5.41958e-07 Conv_1_Relu_ts[6,9]
   + 1.16363e-09 Conv_1_Relu_ts[6,10] - 2.13281e-11 Conv_1_Relu_ts[6,11]
   + 4.12955e-09 Conv_1_Relu_ts[6,12] - 3.71195e-07 Conv_1_Relu_ts[6,13]
   + 2.2241e-11 Conv_1_Relu_ts[6,14] + 6.37491e-09 Conv_1_Relu_ts[6,15]
   - 1.2731e-06 Conv_2_aggregated_features[6,1]
   - 1.0258e-06 Conv_2_aggregated_features[6,4]
   + 4.06095e-10 Conv_2_aggregated_features[6,6]
   + 3.12011e-07 Conv_2_aggregated_features[6,7]
   - 3.09535e-07 Conv_2_aggregated_features[6,9]
   - 0.00177046051248908 Conv_2_aggregated_features[6,10]
   + 8.54383e-13 Conv_2_aggregated_features[6,11]
   - 3.03582e-06 Conv_2_aggregated_features[6,12]
   - 4.41436e-07 Conv_2_aggregated_features[6,13]
   + 7.0584e-12 Conv_2_aggregated_features[6,14]
   - 9.96128e-05 Conv_2_aggregated_features[6,15] + Conv_2_t[6,3]
   = 4.02827e-07
 Conv_2_output_constraint[6,4]: 5.03273e-07 Conv_1_Relu_ts[6,1]
   - 6.22027e-09 Conv_1_Relu_ts[6,5] - 1.47914e-11 Conv_1_Relu_ts[6,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[6,9]
   + 1.18025e-13 Conv_1_Relu_ts[6,10] + 9.23352e-05 Conv_1_Relu_ts[6,11]
   + 0.000518991 Conv_1_Relu_ts[6,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[6,14]
   - 1.05921e-06 Conv_2_aggregated_features[6,1]
   + 2.88866e-10 Conv_2_aggregated_features[6,5]
   - 2.58968e-11 Conv_2_aggregated_features[6,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[6,9]
   - 3.93262e-05 Conv_2_aggregated_features[6,11]
   + 0.000690713 Conv_2_aggregated_features[6,13]
   + 7.70641e-05 Conv_2_aggregated_features[6,14] + Conv_2_t[6,4]
   = -5.84568e-07
 Conv_2_output_constraint[6,5]: 4.37405e-10 Conv_1_Relu_ts[6,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[6,4]
   + 1.77244e-09 Conv_1_Relu_ts[6,5] + 6.99595e-10 Conv_1_Relu_ts[6,9]
   - 1.77001e-13 Conv_1_Relu_ts[6,11] + 2.86208e-12 Conv_1_Relu_ts[6,12]
   + 3.24683e-10 Conv_1_Relu_ts[6,13] - 6.56311e-06 Conv_1_Relu_ts[6,15]
   - 4.63786e-09 Conv_2_aggregated_features[6,0]
   - 4.27797e-10 Conv_2_aggregated_features[6,1]
   - 2.50277e-13 Conv_2_aggregated_features[6,4]
   - 2.35888e-05 Conv_2_aggregated_features[6,5]
   + 6.12624e-10 Conv_2_aggregated_features[6,9]
   + 3.98546e-10 Conv_2_aggregated_features[6,13]
   + 9.89818e-10 Conv_2_aggregated_features[6,15] + Conv_2_t[6,5]
   = 6.24238e-10
 Conv_2_output_constraint[6,6]: - 4.80371e-09 Conv_1_Relu_ts[6,0]
   + 0.2580605149269104 Conv_1_Relu_ts[6,1]
   + 6.84534e-09 Conv_1_Relu_ts[6,2] + 7.44651e-09 Conv_1_Relu_ts[6,3]
   - 2.26729e-08 Conv_1_Relu_ts[6,4] + 1.08868e-09 Conv_1_Relu_ts[6,5]
   + 0.000151346 Conv_1_Relu_ts[6,6] - 1.07095e-09 Conv_1_Relu_ts[6,7]
   - 0.4001754522323608 Conv_1_Relu_ts[6,9]
   - 6.82816e-07 Conv_1_Relu_ts[6,10] - 3.7591e-05 Conv_1_Relu_ts[6,11]
   - 1.62843e-08 Conv_1_Relu_ts[6,12]
   + 1.447632431983948 Conv_1_Relu_ts[6,13]
   - 1.73427e-05 Conv_1_Relu_ts[6,14] + 2.28396e-09 Conv_1_Relu_ts[6,15]
   - 5.33024e-08 Conv_2_aggregated_features[6,0]
   + 0.8234332203865051 Conv_2_aggregated_features[6,1]
   + 6.68388e-09 Conv_2_aggregated_features[6,2]
   - 4.97866e-07 Conv_2_aggregated_features[6,3]
   - 2.61287e-08 Conv_2_aggregated_features[6,4]
   + 5.03108e-11 Conv_2_aggregated_features[6,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[6,6]
   - 1.69808e-09 Conv_2_aggregated_features[6,7]
   - 1.210363864898682 Conv_2_aggregated_features[6,9]
   - 7.75092e-07 Conv_2_aggregated_features[6,10]
   - 6.16735e-05 Conv_2_aggregated_features[6,11]
   - 1.9235e-08 Conv_2_aggregated_features[6,12]
   + 0.4373943507671356 Conv_2_aggregated_features[6,13]
   - 0.00871682073920965 Conv_2_aggregated_features[6,14]
   + 1.60146e-09 Conv_2_aggregated_features[6,15] + Conv_2_t[6,6]
   = 0.1640831530094147
 Conv_2_output_constraint[6,7]: 0.1483074277639389 Conv_1_Relu_ts[6,1]
   + 1.25136e-09 Conv_1_Relu_ts[6,2] - 3.02443e-09 Conv_1_Relu_ts[6,4]
   + 1.45591e-10 Conv_1_Relu_ts[6,5] - 3.6559e-05 Conv_1_Relu_ts[6,6]
   - 7.95116e-08 Conv_1_Relu_ts[6,7]
   + 0.0743252262473106 Conv_1_Relu_ts[6,9]
   + 2.19629e-08 Conv_1_Relu_ts[6,10] - 1.95369e-05 Conv_1_Relu_ts[6,11]
   - 1.21901e-08 Conv_1_Relu_ts[6,12]
   - 0.3415087759494781 Conv_1_Relu_ts[6,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[6,14]
   - 3.54178e-10 Conv_1_Relu_ts[6,15]
   + 0.2313222438097 Conv_2_aggregated_features[6,1]
   + 1.07351e-09 Conv_2_aggregated_features[6,2]
   - 4.05873e-09 Conv_2_aggregated_features[6,4]
   + 1.25187e-10 Conv_2_aggregated_features[6,5]
   - 3.00173e-05 Conv_2_aggregated_features[6,6]
   - 6.57874e-08 Conv_2_aggregated_features[6,7]
   + 0.0370698347687721 Conv_2_aggregated_features[6,9]
   - 2.2615e-08 Conv_2_aggregated_features[6,10]
   - 2.0281e-05 Conv_2_aggregated_features[6,11]
   - 1.14356e-08 Conv_2_aggregated_features[6,12]
   - 0.7083228826522827 Conv_2_aggregated_features[6,13]
   + 0.00042985 Conv_2_aggregated_features[6,14]
   - 4.8015e-10 Conv_2_aggregated_features[6,15] + Conv_2_t[6,7]
   = -0.5274350643157959
 Conv_2_output_constraint[6,8]: - 3.50628e-08 Conv_1_Relu_ts[6,1]
   - 0.00127368199173361 Conv_1_Relu_ts[6,5]
   + 1.1216e-10 Conv_1_Relu_ts[6,6] + 9.94055e-06 Conv_1_Relu_ts[6,9]
   + 2.4753e-05 Conv_1_Relu_ts[6,11] + 2.36679e-06 Conv_1_Relu_ts[6,13]
   + 1.90767e-09 Conv_1_Relu_ts[6,14] - 1.56108e-07 Conv_1_Relu_ts[6,15]
   - 5.18578e-06 Conv_2_aggregated_features[6,1]
   - 2.22254e-05 Conv_2_aggregated_features[6,5]
   + 9.48265e-11 Conv_2_aggregated_features[6,6]
   + 3.14595e-06 Conv_2_aggregated_features[6,9]
   - 1.16025e-05 Conv_2_aggregated_features[6,11]
   + 5.35568e-06 Conv_2_aggregated_features[6,13]
   + 1.47492e-09 Conv_2_aggregated_features[6,14]
   + 1.78245e-06 Conv_2_aggregated_features[6,15] + Conv_2_t[6,8]
   = 1.33952e-09
 Conv_2_output_constraint[6,9]: - 3.05926e-05 Conv_1_Relu_ts[6,1]
   - 4.69457e-10 Conv_1_Relu_ts[6,3] - 1.88994e-07 Conv_1_Relu_ts[6,5]
   + 1.14687e-08 Conv_1_Relu_ts[6,6] - 2.26795e-08 Conv_1_Relu_ts[6,7]
   - 1.1875e-05 Conv_1_Relu_ts[6,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[6,10]
   + 1.3279e-05 Conv_1_Relu_ts[6,11] - 8.03594e-13 Conv_1_Relu_ts[6,12]
   - 1.23268e-05 Conv_1_Relu_ts[6,13] - 1.4779e-07 Conv_1_Relu_ts[6,14]
   - 3.81296e-05 Conv_1_Relu_ts[6,15]
   + 3.05076e-11 Conv_2_aggregated_features[6,0]
   - 1.09971e-05 Conv_2_aggregated_features[6,1]
   - 6.1268e-09 Conv_2_aggregated_features[6,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[6,5]
   + 3.70939e-09 Conv_2_aggregated_features[6,6]
   - 2.02588e-06 Conv_2_aggregated_features[6,7]
   - 1.22094e-05 Conv_2_aggregated_features[6,9]
   - 1.30108e-06 Conv_2_aggregated_features[6,10]
   + 1.18736e-05 Conv_2_aggregated_features[6,11]
   + 9.38886e-13 Conv_2_aggregated_features[6,12]
   - 1.31367e-05 Conv_2_aggregated_features[6,13]
   + 1.5916e-09 Conv_2_aggregated_features[6,14]
   + 5.00944e-10 Conv_2_aggregated_features[6,15] + Conv_2_t[6,9]
   = 2.10696e-05
 Conv_2_output_constraint[6,10]: 2.28334e-09 Conv_1_Relu_ts[6,1]
   - 1.39061e-09 Conv_1_Relu_ts[6,6] + 1.19969e-08 Conv_1_Relu_ts[6,9]
   - 5.35695e-08 Conv_1_Relu_ts[6,10] + 7.265e-09 Conv_1_Relu_ts[6,13]
   - 2.17292e-11 Conv_1_Relu_ts[6,14] + 1.05271e-10 Conv_1_Relu_ts[6,15]
   + 4.90141e-09 Conv_2_aggregated_features[6,1]
   + 6.88e-06 Conv_2_aggregated_features[6,6]
   + 6.7107e-09 Conv_2_aggregated_features[6,9]
   + 1.08045e-08 Conv_2_aggregated_features[6,13]
   - 1.00381e-11 Conv_2_aggregated_features[6,14]
   - 5.27939e-06 Conv_2_aggregated_features[6,15] + Conv_2_t[6,10]
   = -6.36506e-09
 Conv_2_output_constraint[6,11]: - 9.57817e-09 Conv_1_Relu_ts[6,1]
   + 2.02195e-07 Conv_1_Relu_ts[6,2] - 3.79923e-06 Conv_1_Relu_ts[6,5]
   + 5.56008e-05 Conv_1_Relu_ts[6,6] - 5.0366e-09 Conv_1_Relu_ts[6,9]
   - 7.52285e-09 Conv_1_Relu_ts[6,13] - 5.6608e-10 Conv_1_Relu_ts[6,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[6,15]
   - 7.76309e-09 Conv_2_aggregated_features[6,1]
   - 2.62087e-08 Conv_2_aggregated_features[6,2]
   - 3.59639e-06 Conv_2_aggregated_features[6,5]
   - 2.75699e-10 Conv_2_aggregated_features[6,6]
   - 5.01117e-09 Conv_2_aggregated_features[6,9]
   - 7.1903e-09 Conv_2_aggregated_features[6,13]
   - 1.46623e-10 Conv_2_aggregated_features[6,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[6,15]
   + Conv_2_t[6,11] = 3.19541e-09
 Conv_2_output_constraint[6,12]: 1.20473e-05 Conv_1_Relu_ts[6,0]
   + 0.240223228931427 Conv_1_Relu_ts[6,1] + 1.328e-09 Conv_1_Relu_ts[6,2]
   + 2.75965e-08 Conv_1_Relu_ts[6,4] + 1.3558e-09 Conv_1_Relu_ts[6,5]
   - 1.34848e-06 Conv_1_Relu_ts[6,6] + 4.96526e-10 Conv_1_Relu_ts[6,7]
   - 0.3355287313461304 Conv_1_Relu_ts[6,9]
   + 2.31291e-07 Conv_1_Relu_ts[6,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[6,11]
   + 1.69037e-09 Conv_1_Relu_ts[6,12]
   - 0.1011878773570061 Conv_1_Relu_ts[6,13]
   - 0.0104798665270209 Conv_1_Relu_ts[6,14]
   - 4.11824e-11 Conv_1_Relu_ts[6,15]
   + 1.48158e-05 Conv_2_aggregated_features[6,0]
   - 1.069679498672485 Conv_2_aggregated_features[6,1]
   + 1.35995e-09 Conv_2_aggregated_features[6,2]
   + 2.54971e-08 Conv_2_aggregated_features[6,4]
   + 1.4479e-09 Conv_2_aggregated_features[6,5]
   - 2.57874e-05 Conv_2_aggregated_features[6,6]
   + 5.82765e-10 Conv_2_aggregated_features[6,7]
   + 0.2310869246721268 Conv_2_aggregated_features[6,9]
   + 2.42688e-07 Conv_2_aggregated_features[6,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[6,11]
   + 1.80254e-09 Conv_2_aggregated_features[6,12]
   + 0.1041423305869102 Conv_2_aggregated_features[6,13]
   - 0.00980580504983664 Conv_2_aggregated_features[6,14]
   + 4.55897e-11 Conv_2_aggregated_features[6,15] + Conv_2_t[6,12]
   = -0.0243597943335772
 Conv_2_output_constraint[6,13]: - 6.73028e-05 Conv_1_Relu_ts[6,1]
   + 6.15243e-11 Conv_1_Relu_ts[6,2] + 7.12143e-10 Conv_1_Relu_ts[6,4]
   - 3.05256e-08 Conv_1_Relu_ts[6,6]
   + 0.0179135110229254 Conv_1_Relu_ts[6,9]
   - 1.78315e-08 Conv_1_Relu_ts[6,10] + 3.95661e-08 Conv_1_Relu_ts[6,11]
   + 4.14048e-11 Conv_1_Relu_ts[6,12]
   - 0.00573352118954062 Conv_1_Relu_ts[6,13]
   - 1.5325e-05 Conv_1_Relu_ts[6,14]
   - 0.0551034025847912 Conv_2_aggregated_features[6,1]
   + 6.93423e-11 Conv_2_aggregated_features[6,2]
   + 2.18846e-10 Conv_2_aggregated_features[6,4]
   + 1.23153e-11 Conv_2_aggregated_features[6,5]
   - 9.5689e-08 Conv_2_aggregated_features[6,6]
   + 1.83226e-11 Conv_2_aggregated_features[6,7]
   + 0.0415055826306343 Conv_2_aggregated_features[6,9]
   - 4.74355e-09 Conv_2_aggregated_features[6,10]
   - 2.92725e-08 Conv_2_aggregated_features[6,11]
   + 3.37044e-11 Conv_2_aggregated_features[6,12]
   + 0.0228109527379274 Conv_2_aggregated_features[6,13]
   - 3.29092e-05 Conv_2_aggregated_features[6,14]
   - 2.71861e-12 Conv_2_aggregated_features[6,15] + Conv_2_t[6,13]
   = -0.0283597465604544
 Conv_2_output_constraint[6,14]: - 0.3400861024856567 Conv_1_Relu_ts[6,1]
   - 2.59888e-09 Conv_1_Relu_ts[6,2] + 1.10746e-09 Conv_1_Relu_ts[6,4]
   - 4.05832e-10 Conv_1_Relu_ts[6,5] - 2.28438e-05 Conv_1_Relu_ts[6,6]
   - 7.66288e-10 Conv_1_Relu_ts[6,7]
   - 0.1812356412410736 Conv_1_Relu_ts[6,9]
   + 2.36737e-07 Conv_1_Relu_ts[6,10] + 5.46304e-06 Conv_1_Relu_ts[6,11]
   - 1.05702e-08 Conv_1_Relu_ts[6,12]
   - 0.1597411930561066 Conv_1_Relu_ts[6,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[6,14]
   - 1.23647e-09 Conv_1_Relu_ts[6,15]
   - 0.0805128961801529 Conv_2_aggregated_features[6,1]
   - 2.67203e-09 Conv_2_aggregated_features[6,2]
   + 4.58926e-09 Conv_2_aggregated_features[6,4]
   - 5.32055e-10 Conv_2_aggregated_features[6,5]
   - 2.77383e-05 Conv_2_aggregated_features[6,6]
   - 5.76852e-10 Conv_2_aggregated_features[6,7]
   - 0.301887035369873 Conv_2_aggregated_features[6,9]
   + 2.91114e-07 Conv_2_aggregated_features[6,10]
   + 9.46222e-06 Conv_2_aggregated_features[6,11]
   - 1.05406e-08 Conv_2_aggregated_features[6,12]
   - 0.3912120759487152 Conv_2_aggregated_features[6,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[6,14]
   - 9.82623e-10 Conv_2_aggregated_features[6,15] + Conv_2_t[6,14]
   = 0.3463285267353058
 Conv_2_output_constraint[6,15]: 2.91582e-06 Conv_1_Relu_ts[6,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[6,2]
   - 6.96981e-13 Conv_1_Relu_ts[6,5] + 1.06195e-10 Conv_1_Relu_ts[6,6]
   + 9.12088e-06 Conv_1_Relu_ts[6,9] - 7.00879e-07 Conv_1_Relu_ts[6,10]
   + 2.54492e-08 Conv_1_Relu_ts[6,11] - 8.8874e-07 Conv_1_Relu_ts[6,12]
   - 9.8182e-06 Conv_1_Relu_ts[6,13] + 1.53315e-07 Conv_1_Relu_ts[6,14]
   - 2.41261e-08 Conv_1_Relu_ts[6,15]
   + 3.09046e-06 Conv_2_aggregated_features[6,1]
   - 5.23654e-07 Conv_2_aggregated_features[6,2]
   + 3.40881e-13 Conv_2_aggregated_features[6,5]
   + 4.84763e-12 Conv_2_aggregated_features[6,6]
   + 6.94651e-06 Conv_2_aggregated_features[6,9]
   + 1.80346e-08 Conv_2_aggregated_features[6,10]
   - 8.89515e-09 Conv_2_aggregated_features[6,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[6,12]
   - 5.75117e-06 Conv_2_aggregated_features[6,13]
   - 3.11447e-07 Conv_2_aggregated_features[6,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[6,15]
   + Conv_2_t[6,15] = -5.06887e-06
 Conv_2_output_constraint[7,0]: 5.62815e-10 Conv_1_Relu_ts[7,7]
   + 1.92775e-11 Conv_1_Relu_ts[7,10]
   + 4.6483942423947155e-04 Conv_1_Relu_ts[7,15]
   + 5.32965e-11 Conv_2_aggregated_features[7,7]
   - 3.50224e-08 Conv_2_aggregated_features[7,15] + Conv_2_t[7,0]
   = 2.0494e-14
 Conv_2_output_constraint[7,1]: - 3.1209e-05 Conv_1_Relu_ts[7,1]
   + 3.2097e-12 Conv_1_Relu_ts[7,2] + 6.75828e-11 Conv_1_Relu_ts[7,4]
   + 6.05596e-11 Conv_1_Relu_ts[7,5] + 9.57194e-07 Conv_1_Relu_ts[7,6]
   - 1.94554e-13 Conv_1_Relu_ts[7,7] + 9.60207e-05 Conv_1_Relu_ts[7,9]
   + 3.54229e-11 Conv_1_Relu_ts[7,10] + 2.0034e-09 Conv_1_Relu_ts[7,11]
   + 3.41256e-05 Conv_1_Relu_ts[7,13] + 2.24541e-06 Conv_1_Relu_ts[7,14]
   - 7.56678e-10 Conv_1_Relu_ts[7,15]
   - 3.63258e-06 Conv_2_aggregated_features[7,1]
   - 2.02372e-12 Conv_2_aggregated_features[7,2]
   - 6.7717e-13 Conv_2_aggregated_features[7,4]
   + 1.64055e-10 Conv_2_aggregated_features[7,5]
   + 8.63851e-05 Conv_2_aggregated_features[7,6]
   - 5.43338e-10 Conv_2_aggregated_features[7,7]
   + 8.87988e-05 Conv_2_aggregated_features[7,9]
   + 2.03334e-11 Conv_2_aggregated_features[7,10]
   + 2.09865e-09 Conv_2_aggregated_features[7,11]
   + 5.1289e-05 Conv_2_aggregated_features[7,13]
   - 1.99215e-06 Conv_2_aggregated_features[7,14]
   - 2.85245e-06 Conv_2_aggregated_features[7,15] + Conv_2_t[7,1]
   = -5.09804e-05
 Conv_2_output_constraint[7,2]: 4.6554e-12 Conv_1_Relu_ts[7,1]
   - 5.08004e-07 Conv_1_Relu_ts[7,4] + 8.2088e-12 Conv_1_Relu_ts[7,9]
   + 3.89405e-05 Conv_1_Relu_ts[7,10] + 3.83811e-12 Conv_1_Relu_ts[7,13]
   - 1.9210424215998501e-04 Conv_1_Relu_ts[7,15]
   + 2.16136e-12 Conv_2_aggregated_features[7,1]
   - 4.18879e-13 Conv_2_aggregated_features[7,4]
   + 1.01588e-11 Conv_2_aggregated_features[7,9]
   + 4.38408e-12 Conv_2_aggregated_features[7,13]
   + 6.90517e-06 Conv_2_aggregated_features[7,15] + Conv_2_t[7,2]
   = -1.21231e-11
 Conv_2_output_constraint[7,3]: - 9.77857e-07 Conv_1_Relu_ts[7,1]
   - 1.62341e-06 Conv_1_Relu_ts[7,4] - 1.26004e-09 Conv_1_Relu_ts[7,6]
   + 2.40003e-07 Conv_1_Relu_ts[7,7] - 5.41958e-07 Conv_1_Relu_ts[7,9]
   + 1.16363e-09 Conv_1_Relu_ts[7,10] - 2.13281e-11 Conv_1_Relu_ts[7,11]
   + 4.12955e-09 Conv_1_Relu_ts[7,12] - 3.71195e-07 Conv_1_Relu_ts[7,13]
   + 2.2241e-11 Conv_1_Relu_ts[7,14] + 6.37491e-09 Conv_1_Relu_ts[7,15]
   - 1.2731e-06 Conv_2_aggregated_features[7,1]
   - 1.0258e-06 Conv_2_aggregated_features[7,4]
   + 4.06095e-10 Conv_2_aggregated_features[7,6]
   + 3.12011e-07 Conv_2_aggregated_features[7,7]
   - 3.09535e-07 Conv_2_aggregated_features[7,9]
   - 0.00177046051248908 Conv_2_aggregated_features[7,10]
   + 8.54383e-13 Conv_2_aggregated_features[7,11]
   - 3.03582e-06 Conv_2_aggregated_features[7,12]
   - 4.41436e-07 Conv_2_aggregated_features[7,13]
   + 7.0584e-12 Conv_2_aggregated_features[7,14]
   - 9.96128e-05 Conv_2_aggregated_features[7,15] + Conv_2_t[7,3]
   = 4.02827e-07
 Conv_2_output_constraint[7,4]: 5.03273e-07 Conv_1_Relu_ts[7,1]
   - 6.22027e-09 Conv_1_Relu_ts[7,5] - 1.47914e-11 Conv_1_Relu_ts[7,6]
   + 2.2143042588140815e-04 Conv_1_Relu_ts[7,9]
   + 1.18025e-13 Conv_1_Relu_ts[7,10] + 9.23352e-05 Conv_1_Relu_ts[7,11]
   + 0.000518991 Conv_1_Relu_ts[7,13]
   + 2.0950080943293869e-04 Conv_1_Relu_ts[7,14]
   - 1.05921e-06 Conv_2_aggregated_features[7,1]
   + 2.88866e-10 Conv_2_aggregated_features[7,5]
   - 2.58968e-11 Conv_2_aggregated_features[7,6]
   + 1.6886646335478872e-04 Conv_2_aggregated_features[7,9]
   - 3.93262e-05 Conv_2_aggregated_features[7,11]
   + 0.000690713 Conv_2_aggregated_features[7,13]
   + 7.70641e-05 Conv_2_aggregated_features[7,14] + Conv_2_t[7,4]
   = -5.84568e-07
 Conv_2_output_constraint[7,5]: 4.37405e-10 Conv_1_Relu_ts[7,1]
   - 4.3527575326152146e-04 Conv_1_Relu_ts[7,4]
   + 1.77244e-09 Conv_1_Relu_ts[7,5] + 6.99595e-10 Conv_1_Relu_ts[7,9]
   - 1.77001e-13 Conv_1_Relu_ts[7,11] + 2.86208e-12 Conv_1_Relu_ts[7,12]
   + 3.24683e-10 Conv_1_Relu_ts[7,13] - 6.56311e-06 Conv_1_Relu_ts[7,15]
   - 4.63786e-09 Conv_2_aggregated_features[7,0]
   - 4.27797e-10 Conv_2_aggregated_features[7,1]
   - 2.50277e-13 Conv_2_aggregated_features[7,4]
   - 2.35888e-05 Conv_2_aggregated_features[7,5]
   + 6.12624e-10 Conv_2_aggregated_features[7,9]
   + 3.98546e-10 Conv_2_aggregated_features[7,13]
   + 9.89818e-10 Conv_2_aggregated_features[7,15] + Conv_2_t[7,5]
   = 6.24238e-10
 Conv_2_output_constraint[7,6]: - 4.80371e-09 Conv_1_Relu_ts[7,0]
   + 0.2580605149269104 Conv_1_Relu_ts[7,1]
   + 6.84534e-09 Conv_1_Relu_ts[7,2] + 7.44651e-09 Conv_1_Relu_ts[7,3]
   - 2.26729e-08 Conv_1_Relu_ts[7,4] + 1.08868e-09 Conv_1_Relu_ts[7,5]
   + 0.000151346 Conv_1_Relu_ts[7,6] - 1.07095e-09 Conv_1_Relu_ts[7,7]
   - 0.4001754522323608 Conv_1_Relu_ts[7,9]
   - 6.82816e-07 Conv_1_Relu_ts[7,10] - 3.7591e-05 Conv_1_Relu_ts[7,11]
   - 1.62843e-08 Conv_1_Relu_ts[7,12]
   + 1.447632431983948 Conv_1_Relu_ts[7,13]
   - 1.73427e-05 Conv_1_Relu_ts[7,14] + 2.28396e-09 Conv_1_Relu_ts[7,15]
   - 5.33024e-08 Conv_2_aggregated_features[7,0]
   + 0.8234332203865051 Conv_2_aggregated_features[7,1]
   + 6.68388e-09 Conv_2_aggregated_features[7,2]
   - 4.97866e-07 Conv_2_aggregated_features[7,3]
   - 2.61287e-08 Conv_2_aggregated_features[7,4]
   + 5.03108e-11 Conv_2_aggregated_features[7,5]
   + 1.4717654266860336e-04 Conv_2_aggregated_features[7,6]
   - 1.69808e-09 Conv_2_aggregated_features[7,7]
   - 1.210363864898682 Conv_2_aggregated_features[7,9]
   - 7.75092e-07 Conv_2_aggregated_features[7,10]
   - 6.16735e-05 Conv_2_aggregated_features[7,11]
   - 1.9235e-08 Conv_2_aggregated_features[7,12]
   + 0.4373943507671356 Conv_2_aggregated_features[7,13]
   - 0.00871682073920965 Conv_2_aggregated_features[7,14]
   + 1.60146e-09 Conv_2_aggregated_features[7,15] + Conv_2_t[7,6]
   = 0.1640831530094147
 Conv_2_output_constraint[7,7]: 0.1483074277639389 Conv_1_Relu_ts[7,1]
   + 1.25136e-09 Conv_1_Relu_ts[7,2] - 3.02443e-09 Conv_1_Relu_ts[7,4]
   + 1.45591e-10 Conv_1_Relu_ts[7,5] - 3.6559e-05 Conv_1_Relu_ts[7,6]
   - 7.95116e-08 Conv_1_Relu_ts[7,7]
   + 0.0743252262473106 Conv_1_Relu_ts[7,9]
   + 2.19629e-08 Conv_1_Relu_ts[7,10] - 1.95369e-05 Conv_1_Relu_ts[7,11]
   - 1.21901e-08 Conv_1_Relu_ts[7,12]
   - 0.3415087759494781 Conv_1_Relu_ts[7,13]
   + 3.1246477738022804e-04 Conv_1_Relu_ts[7,14]
   - 3.54178e-10 Conv_1_Relu_ts[7,15]
   + 0.2313222438097 Conv_2_aggregated_features[7,1]
   + 1.07351e-09 Conv_2_aggregated_features[7,2]
   - 4.05873e-09 Conv_2_aggregated_features[7,4]
   + 1.25187e-10 Conv_2_aggregated_features[7,5]
   - 3.00173e-05 Conv_2_aggregated_features[7,6]
   - 6.57874e-08 Conv_2_aggregated_features[7,7]
   + 0.0370698347687721 Conv_2_aggregated_features[7,9]
   - 2.2615e-08 Conv_2_aggregated_features[7,10]
   - 2.0281e-05 Conv_2_aggregated_features[7,11]
   - 1.14356e-08 Conv_2_aggregated_features[7,12]
   - 0.7083228826522827 Conv_2_aggregated_features[7,13]
   + 0.00042985 Conv_2_aggregated_features[7,14]
   - 4.8015e-10 Conv_2_aggregated_features[7,15] + Conv_2_t[7,7]
   = -0.5274350643157959
 Conv_2_output_constraint[7,8]: - 3.50628e-08 Conv_1_Relu_ts[7,1]
   - 0.00127368199173361 Conv_1_Relu_ts[7,5]
   + 1.1216e-10 Conv_1_Relu_ts[7,6] + 9.94055e-06 Conv_1_Relu_ts[7,9]
   + 2.4753e-05 Conv_1_Relu_ts[7,11] + 2.36679e-06 Conv_1_Relu_ts[7,13]
   + 1.90767e-09 Conv_1_Relu_ts[7,14] - 1.56108e-07 Conv_1_Relu_ts[7,15]
   - 5.18578e-06 Conv_2_aggregated_features[7,1]
   - 2.22254e-05 Conv_2_aggregated_features[7,5]
   + 9.48265e-11 Conv_2_aggregated_features[7,6]
   + 3.14595e-06 Conv_2_aggregated_features[7,9]
   - 1.16025e-05 Conv_2_aggregated_features[7,11]
   + 5.35568e-06 Conv_2_aggregated_features[7,13]
   + 1.47492e-09 Conv_2_aggregated_features[7,14]
   + 1.78245e-06 Conv_2_aggregated_features[7,15] + Conv_2_t[7,8]
   = 1.33952e-09
 Conv_2_output_constraint[7,9]: - 3.05926e-05 Conv_1_Relu_ts[7,1]
   - 4.69457e-10 Conv_1_Relu_ts[7,3] - 1.88994e-07 Conv_1_Relu_ts[7,5]
   + 1.14687e-08 Conv_1_Relu_ts[7,6] - 2.26795e-08 Conv_1_Relu_ts[7,7]
   - 1.1875e-05 Conv_1_Relu_ts[7,9]
   + 1.6695528756827116e-04 Conv_1_Relu_ts[7,10]
   + 1.3279e-05 Conv_1_Relu_ts[7,11] - 8.03594e-13 Conv_1_Relu_ts[7,12]
   - 1.23268e-05 Conv_1_Relu_ts[7,13] - 1.4779e-07 Conv_1_Relu_ts[7,14]
   - 3.81296e-05 Conv_1_Relu_ts[7,15]
   + 3.05076e-11 Conv_2_aggregated_features[7,0]
   - 1.09971e-05 Conv_2_aggregated_features[7,1]
   - 6.1268e-09 Conv_2_aggregated_features[7,3]
   - 9.4766262918710709e-04 Conv_2_aggregated_features[7,5]
   + 3.70939e-09 Conv_2_aggregated_features[7,6]
   - 2.02588e-06 Conv_2_aggregated_features[7,7]
   - 1.22094e-05 Conv_2_aggregated_features[7,9]
   - 1.30108e-06 Conv_2_aggregated_features[7,10]
   + 1.18736e-05 Conv_2_aggregated_features[7,11]
   + 9.38886e-13 Conv_2_aggregated_features[7,12]
   - 1.31367e-05 Conv_2_aggregated_features[7,13]
   + 1.5916e-09 Conv_2_aggregated_features[7,14]
   + 5.00944e-10 Conv_2_aggregated_features[7,15] + Conv_2_t[7,9]
   = 2.10696e-05
 Conv_2_output_constraint[7,10]: 2.28334e-09 Conv_1_Relu_ts[7,1]
   - 1.39061e-09 Conv_1_Relu_ts[7,6] + 1.19969e-08 Conv_1_Relu_ts[7,9]
   - 5.35695e-08 Conv_1_Relu_ts[7,10] + 7.265e-09 Conv_1_Relu_ts[7,13]
   - 2.17292e-11 Conv_1_Relu_ts[7,14] + 1.05271e-10 Conv_1_Relu_ts[7,15]
   + 4.90141e-09 Conv_2_aggregated_features[7,1]
   + 6.88e-06 Conv_2_aggregated_features[7,6]
   + 6.7107e-09 Conv_2_aggregated_features[7,9]
   + 1.08045e-08 Conv_2_aggregated_features[7,13]
   - 1.00381e-11 Conv_2_aggregated_features[7,14]
   - 5.27939e-06 Conv_2_aggregated_features[7,15] + Conv_2_t[7,10]
   = -6.36506e-09
 Conv_2_output_constraint[7,11]: - 9.57817e-09 Conv_1_Relu_ts[7,1]
   + 2.02195e-07 Conv_1_Relu_ts[7,2] - 3.79923e-06 Conv_1_Relu_ts[7,5]
   + 5.56008e-05 Conv_1_Relu_ts[7,6] - 5.0366e-09 Conv_1_Relu_ts[7,9]
   - 7.52285e-09 Conv_1_Relu_ts[7,13] - 5.6608e-10 Conv_1_Relu_ts[7,14]
   - 4.3281077523715794e-04 Conv_1_Relu_ts[7,15]
   - 7.76309e-09 Conv_2_aggregated_features[7,1]
   - 2.62087e-08 Conv_2_aggregated_features[7,2]
   - 3.59639e-06 Conv_2_aggregated_features[7,5]
   - 2.75699e-10 Conv_2_aggregated_features[7,6]
   - 5.01117e-09 Conv_2_aggregated_features[7,9]
   - 7.1903e-09 Conv_2_aggregated_features[7,13]
   - 1.46623e-10 Conv_2_aggregated_features[7,14]
   + 3.3513037487864494e-04 Conv_2_aggregated_features[7,15]
   + Conv_2_t[7,11] = 3.19541e-09
 Conv_2_output_constraint[7,12]: 1.20473e-05 Conv_1_Relu_ts[7,0]
   + 0.240223228931427 Conv_1_Relu_ts[7,1] + 1.328e-09 Conv_1_Relu_ts[7,2]
   + 2.75965e-08 Conv_1_Relu_ts[7,4] + 1.3558e-09 Conv_1_Relu_ts[7,5]
   - 1.34848e-06 Conv_1_Relu_ts[7,6] + 4.96526e-10 Conv_1_Relu_ts[7,7]
   - 0.3355287313461304 Conv_1_Relu_ts[7,9]
   + 2.31291e-07 Conv_1_Relu_ts[7,10]
   + 1.1646050552371889e-04 Conv_1_Relu_ts[7,11]
   + 1.69037e-09 Conv_1_Relu_ts[7,12]
   - 0.1011878773570061 Conv_1_Relu_ts[7,13]
   - 0.0104798665270209 Conv_1_Relu_ts[7,14]
   - 4.11824e-11 Conv_1_Relu_ts[7,15]
   + 1.48158e-05 Conv_2_aggregated_features[7,0]
   - 1.069679498672485 Conv_2_aggregated_features[7,1]
   + 1.35995e-09 Conv_2_aggregated_features[7,2]
   + 2.54971e-08 Conv_2_aggregated_features[7,4]
   + 1.4479e-09 Conv_2_aggregated_features[7,5]
   - 2.57874e-05 Conv_2_aggregated_features[7,6]
   + 5.82765e-10 Conv_2_aggregated_features[7,7]
   + 0.2310869246721268 Conv_2_aggregated_features[7,9]
   + 2.42688e-07 Conv_2_aggregated_features[7,10]
   + 1.2337348016444594e-04 Conv_2_aggregated_features[7,11]
   + 1.80254e-09 Conv_2_aggregated_features[7,12]
   + 0.1041423305869102 Conv_2_aggregated_features[7,13]
   - 0.00980580504983664 Conv_2_aggregated_features[7,14]
   + 4.55897e-11 Conv_2_aggregated_features[7,15] + Conv_2_t[7,12]
   = -0.0243597943335772
 Conv_2_output_constraint[7,13]: - 6.73028e-05 Conv_1_Relu_ts[7,1]
   + 6.15243e-11 Conv_1_Relu_ts[7,2] + 7.12143e-10 Conv_1_Relu_ts[7,4]
   - 3.05256e-08 Conv_1_Relu_ts[7,6]
   + 0.0179135110229254 Conv_1_Relu_ts[7,9]
   - 1.78315e-08 Conv_1_Relu_ts[7,10] + 3.95661e-08 Conv_1_Relu_ts[7,11]
   + 4.14048e-11 Conv_1_Relu_ts[7,12]
   - 0.00573352118954062 Conv_1_Relu_ts[7,13]
   - 1.5325e-05 Conv_1_Relu_ts[7,14]
   - 0.0551034025847912 Conv_2_aggregated_features[7,1]
   + 6.93423e-11 Conv_2_aggregated_features[7,2]
   + 2.18846e-10 Conv_2_aggregated_features[7,4]
   + 1.23153e-11 Conv_2_aggregated_features[7,5]
   - 9.5689e-08 Conv_2_aggregated_features[7,6]
   + 1.83226e-11 Conv_2_aggregated_features[7,7]
   + 0.0415055826306343 Conv_2_aggregated_features[7,9]
   - 4.74355e-09 Conv_2_aggregated_features[7,10]
   - 2.92725e-08 Conv_2_aggregated_features[7,11]
   + 3.37044e-11 Conv_2_aggregated_features[7,12]
   + 0.0228109527379274 Conv_2_aggregated_features[7,13]
   - 3.29092e-05 Conv_2_aggregated_features[7,14]
   - 2.71861e-12 Conv_2_aggregated_features[7,15] + Conv_2_t[7,13]
   = -0.0283597465604544
 Conv_2_output_constraint[7,14]: - 0.3400861024856567 Conv_1_Relu_ts[7,1]
   - 2.59888e-09 Conv_1_Relu_ts[7,2] + 1.10746e-09 Conv_1_Relu_ts[7,4]
   - 4.05832e-10 Conv_1_Relu_ts[7,5] - 2.28438e-05 Conv_1_Relu_ts[7,6]
   - 7.66288e-10 Conv_1_Relu_ts[7,7]
   - 0.1812356412410736 Conv_1_Relu_ts[7,9]
   + 2.36737e-07 Conv_1_Relu_ts[7,10] + 5.46304e-06 Conv_1_Relu_ts[7,11]
   - 1.05702e-08 Conv_1_Relu_ts[7,12]
   - 0.1597411930561066 Conv_1_Relu_ts[7,13]
   - 6.5691082272678614e-04 Conv_1_Relu_ts[7,14]
   - 1.23647e-09 Conv_1_Relu_ts[7,15]
   - 0.0805128961801529 Conv_2_aggregated_features[7,1]
   - 2.67203e-09 Conv_2_aggregated_features[7,2]
   + 4.58926e-09 Conv_2_aggregated_features[7,4]
   - 5.32055e-10 Conv_2_aggregated_features[7,5]
   - 2.77383e-05 Conv_2_aggregated_features[7,6]
   - 5.76852e-10 Conv_2_aggregated_features[7,7]
   - 0.301887035369873 Conv_2_aggregated_features[7,9]
   + 2.91114e-07 Conv_2_aggregated_features[7,10]
   + 9.46222e-06 Conv_2_aggregated_features[7,11]
   - 1.05406e-08 Conv_2_aggregated_features[7,12]
   - 0.3912120759487152 Conv_2_aggregated_features[7,13]
   - 5.1420519594103098e-04 Conv_2_aggregated_features[7,14]
   - 9.82623e-10 Conv_2_aggregated_features[7,15] + Conv_2_t[7,14]
   = 0.3463285267353058
 Conv_2_output_constraint[7,15]: 2.91582e-06 Conv_1_Relu_ts[7,1]
   - 1.1535656085470691e-04 Conv_1_Relu_ts[7,2]
   - 6.96981e-13 Conv_1_Relu_ts[7,5] + 1.06195e-10 Conv_1_Relu_ts[7,6]
   + 9.12088e-06 Conv_1_Relu_ts[7,9] - 7.00879e-07 Conv_1_Relu_ts[7,10]
   + 2.54492e-08 Conv_1_Relu_ts[7,11] - 8.8874e-07 Conv_1_Relu_ts[7,12]
   - 9.8182e-06 Conv_1_Relu_ts[7,13] + 1.53315e-07 Conv_1_Relu_ts[7,14]
   - 2.41261e-08 Conv_1_Relu_ts[7,15]
   + 3.09046e-06 Conv_2_aggregated_features[7,1]
   - 5.23654e-07 Conv_2_aggregated_features[7,2]
   + 3.40881e-13 Conv_2_aggregated_features[7,5]
   + 4.84763e-12 Conv_2_aggregated_features[7,6]
   + 6.94651e-06 Conv_2_aggregated_features[7,9]
   + 1.80346e-08 Conv_2_aggregated_features[7,10]
   - 8.89515e-09 Conv_2_aggregated_features[7,11]
   + 2.0948774181306362e-04 Conv_2_aggregated_features[7,12]
   - 5.75117e-06 Conv_2_aggregated_features[7,13]
   - 3.11447e-07 Conv_2_aggregated_features[7,14]
   - 4.9647950800135732e-04 Conv_2_aggregated_features[7,15]
   + Conv_2_t[7,15] = -5.06887e-06
 Conv_2_Relu_constraint_1[0,0]: - Conv_2_t[0,0] + Conv_2_Relu_ts[0,0]
   - Conv_2_Relu_ss[0,0] = 0
 Conv_2_Relu_constraint_1[0,1]: - Conv_2_t[0,1] + Conv_2_Relu_ts[0,1]
   - Conv_2_Relu_ss[0,1] = 0
 Conv_2_Relu_constraint_1[0,2]: - Conv_2_t[0,2] + Conv_2_Relu_ts[0,2]
   - Conv_2_Relu_ss[0,2] = 0
 Conv_2_Relu_constraint_1[0,3]: - Conv_2_t[0,3] + Conv_2_Relu_ts[0,3]
   - Conv_2_Relu_ss[0,3] = 0
 Conv_2_Relu_constraint_1[0,4]: - Conv_2_t[0,4] + Conv_2_Relu_ts[0,4]
   - Conv_2_Relu_ss[0,4] = 0
 Conv_2_Relu_constraint_1[0,5]: - Conv_2_t[0,5] + Conv_2_Relu_ts[0,5]
   - Conv_2_Relu_ss[0,5] = 0
 Conv_2_Relu_constraint_1[0,6]: - Conv_2_t[0,6] + Conv_2_Relu_ts[0,6]
   - Conv_2_Relu_ss[0,6] = 0
 Conv_2_Relu_constraint_1[0,7]: - Conv_2_t[0,7] + Conv_2_Relu_ts[0,7]
   - Conv_2_Relu_ss[0,7] = 0
 Conv_2_Relu_constraint_1[0,8]: - Conv_2_t[0,8] + Conv_2_Relu_ts[0,8]
   - Conv_2_Relu_ss[0,8] = 0
 Conv_2_Relu_constraint_1[0,9]: - Conv_2_t[0,9] + Conv_2_Relu_ts[0,9]
   - Conv_2_Relu_ss[0,9] = 0
 Conv_2_Relu_constraint_1[0,10]: - Conv_2_t[0,10] + Conv_2_Relu_ts[0,10]
   - Conv_2_Relu_ss[0,10] = 0
 Conv_2_Relu_constraint_1[0,11]: - Conv_2_t[0,11] + Conv_2_Relu_ts[0,11]
   - Conv_2_Relu_ss[0,11] = 0
 Conv_2_Relu_constraint_1[0,12]: - Conv_2_t[0,12] + Conv_2_Relu_ts[0,12]
   - Conv_2_Relu_ss[0,12] = 0
 Conv_2_Relu_constraint_1[0,13]: - Conv_2_t[0,13] + Conv_2_Relu_ts[0,13]
   - Conv_2_Relu_ss[0,13] = 0
 Conv_2_Relu_constraint_1[0,14]: - Conv_2_t[0,14] + Conv_2_Relu_ts[0,14]
   - Conv_2_Relu_ss[0,14] = 0
 Conv_2_Relu_constraint_1[0,15]: - Conv_2_t[0,15] + Conv_2_Relu_ts[0,15]
   - Conv_2_Relu_ss[0,15] = 0
 Conv_2_Relu_constraint_1[1,0]: - Conv_2_t[1,0] + Conv_2_Relu_ts[1,0]
   - Conv_2_Relu_ss[1,0] = 0
 Conv_2_Relu_constraint_1[1,1]: - Conv_2_t[1,1] + Conv_2_Relu_ts[1,1]
   - Conv_2_Relu_ss[1,1] = 0
 Conv_2_Relu_constraint_1[1,2]: - Conv_2_t[1,2] + Conv_2_Relu_ts[1,2]
   - Conv_2_Relu_ss[1,2] = 0
 Conv_2_Relu_constraint_1[1,3]: - Conv_2_t[1,3] + Conv_2_Relu_ts[1,3]
   - Conv_2_Relu_ss[1,3] = 0
 Conv_2_Relu_constraint_1[1,4]: - Conv_2_t[1,4] + Conv_2_Relu_ts[1,4]
   - Conv_2_Relu_ss[1,4] = 0
 Conv_2_Relu_constraint_1[1,5]: - Conv_2_t[1,5] + Conv_2_Relu_ts[1,5]
   - Conv_2_Relu_ss[1,5] = 0
 Conv_2_Relu_constraint_1[1,6]: - Conv_2_t[1,6] + Conv_2_Relu_ts[1,6]
   - Conv_2_Relu_ss[1,6] = 0
 Conv_2_Relu_constraint_1[1,7]: - Conv_2_t[1,7] + Conv_2_Relu_ts[1,7]
   - Conv_2_Relu_ss[1,7] = 0
 Conv_2_Relu_constraint_1[1,8]: - Conv_2_t[1,8] + Conv_2_Relu_ts[1,8]
   - Conv_2_Relu_ss[1,8] = 0
 Conv_2_Relu_constraint_1[1,9]: - Conv_2_t[1,9] + Conv_2_Relu_ts[1,9]
   - Conv_2_Relu_ss[1,9] = 0
 Conv_2_Relu_constraint_1[1,10]: - Conv_2_t[1,10] + Conv_2_Relu_ts[1,10]
   - Conv_2_Relu_ss[1,10] = 0
 Conv_2_Relu_constraint_1[1,11]: - Conv_2_t[1,11] + Conv_2_Relu_ts[1,11]
   - Conv_2_Relu_ss[1,11] = 0
 Conv_2_Relu_constraint_1[1,12]: - Conv_2_t[1,12] + Conv_2_Relu_ts[1,12]
   - Conv_2_Relu_ss[1,12] = 0
 Conv_2_Relu_constraint_1[1,13]: - Conv_2_t[1,13] + Conv_2_Relu_ts[1,13]
   - Conv_2_Relu_ss[1,13] = 0
 Conv_2_Relu_constraint_1[1,14]: - Conv_2_t[1,14] + Conv_2_Relu_ts[1,14]
   - Conv_2_Relu_ss[1,14] = 0
 Conv_2_Relu_constraint_1[1,15]: - Conv_2_t[1,15] + Conv_2_Relu_ts[1,15]
   - Conv_2_Relu_ss[1,15] = 0
 Conv_2_Relu_constraint_1[2,0]: - Conv_2_t[2,0] + Conv_2_Relu_ts[2,0]
   - Conv_2_Relu_ss[2,0] = 0
 Conv_2_Relu_constraint_1[2,1]: - Conv_2_t[2,1] + Conv_2_Relu_ts[2,1]
   - Conv_2_Relu_ss[2,1] = 0
 Conv_2_Relu_constraint_1[2,2]: - Conv_2_t[2,2] + Conv_2_Relu_ts[2,2]
   - Conv_2_Relu_ss[2,2] = 0
 Conv_2_Relu_constraint_1[2,3]: - Conv_2_t[2,3] + Conv_2_Relu_ts[2,3]
   - Conv_2_Relu_ss[2,3] = 0
 Conv_2_Relu_constraint_1[2,4]: - Conv_2_t[2,4] + Conv_2_Relu_ts[2,4]
   - Conv_2_Relu_ss[2,4] = 0
 Conv_2_Relu_constraint_1[2,5]: - Conv_2_t[2,5] + Conv_2_Relu_ts[2,5]
   - Conv_2_Relu_ss[2,5] = 0
 Conv_2_Relu_constraint_1[2,6]: - Conv_2_t[2,6] + Conv_2_Relu_ts[2,6]
   - Conv_2_Relu_ss[2,6] = 0
 Conv_2_Relu_constraint_1[2,7]: - Conv_2_t[2,7] + Conv_2_Relu_ts[2,7]
   - Conv_2_Relu_ss[2,7] = 0
 Conv_2_Relu_constraint_1[2,8]: - Conv_2_t[2,8] + Conv_2_Relu_ts[2,8]
   - Conv_2_Relu_ss[2,8] = 0
 Conv_2_Relu_constraint_1[2,9]: - Conv_2_t[2,9] + Conv_2_Relu_ts[2,9]
   - Conv_2_Relu_ss[2,9] = 0
 Conv_2_Relu_constraint_1[2,10]: - Conv_2_t[2,10] + Conv_2_Relu_ts[2,10]
   - Conv_2_Relu_ss[2,10] = 0
 Conv_2_Relu_constraint_1[2,11]: - Conv_2_t[2,11] + Conv_2_Relu_ts[2,11]
   - Conv_2_Relu_ss[2,11] = 0
 Conv_2_Relu_constraint_1[2,12]: - Conv_2_t[2,12] + Conv_2_Relu_ts[2,12]
   - Conv_2_Relu_ss[2,12] = 0
 Conv_2_Relu_constraint_1[2,13]: - Conv_2_t[2,13] + Conv_2_Relu_ts[2,13]
   - Conv_2_Relu_ss[2,13] = 0
 Conv_2_Relu_constraint_1[2,14]: - Conv_2_t[2,14] + Conv_2_Relu_ts[2,14]
   - Conv_2_Relu_ss[2,14] = 0
 Conv_2_Relu_constraint_1[2,15]: - Conv_2_t[2,15] + Conv_2_Relu_ts[2,15]
   - Conv_2_Relu_ss[2,15] = 0
 Conv_2_Relu_constraint_1[3,0]: - Conv_2_t[3,0] + Conv_2_Relu_ts[3,0]
   - Conv_2_Relu_ss[3,0] = 0
 Conv_2_Relu_constraint_1[3,1]: - Conv_2_t[3,1] + Conv_2_Relu_ts[3,1]
   - Conv_2_Relu_ss[3,1] = 0
 Conv_2_Relu_constraint_1[3,2]: - Conv_2_t[3,2] + Conv_2_Relu_ts[3,2]
   - Conv_2_Relu_ss[3,2] = 0
 Conv_2_Relu_constraint_1[3,3]: - Conv_2_t[3,3] + Conv_2_Relu_ts[3,3]
   - Conv_2_Relu_ss[3,3] = 0
 Conv_2_Relu_constraint_1[3,4]: - Conv_2_t[3,4] + Conv_2_Relu_ts[3,4]
   - Conv_2_Relu_ss[3,4] = 0
 Conv_2_Relu_constraint_1[3,5]: - Conv_2_t[3,5] + Conv_2_Relu_ts[3,5]
   - Conv_2_Relu_ss[3,5] = 0
 Conv_2_Relu_constraint_1[3,6]: - Conv_2_t[3,6] + Conv_2_Relu_ts[3,6]
   - Conv_2_Relu_ss[3,6] = 0
 Conv_2_Relu_constraint_1[3,7]: - Conv_2_t[3,7] + Conv_2_Relu_ts[3,7]
   - Conv_2_Relu_ss[3,7] = 0
 Conv_2_Relu_constraint_1[3,8]: - Conv_2_t[3,8] + Conv_2_Relu_ts[3,8]
   - Conv_2_Relu_ss[3,8] = 0
 Conv_2_Relu_constraint_1[3,9]: - Conv_2_t[3,9] + Conv_2_Relu_ts[3,9]
   - Conv_2_Relu_ss[3,9] = 0
 Conv_2_Relu_constraint_1[3,10]: - Conv_2_t[3,10] + Conv_2_Relu_ts[3,10]
   - Conv_2_Relu_ss[3,10] = 0
 Conv_2_Relu_constraint_1[3,11]: - Conv_2_t[3,11] + Conv_2_Relu_ts[3,11]
   - Conv_2_Relu_ss[3,11] = 0
 Conv_2_Relu_constraint_1[3,12]: - Conv_2_t[3,12] + Conv_2_Relu_ts[3,12]
   - Conv_2_Relu_ss[3,12] = 0
 Conv_2_Relu_constraint_1[3,13]: - Conv_2_t[3,13] + Conv_2_Relu_ts[3,13]
   - Conv_2_Relu_ss[3,13] = 0
 Conv_2_Relu_constraint_1[3,14]: - Conv_2_t[3,14] + Conv_2_Relu_ts[3,14]
   - Conv_2_Relu_ss[3,14] = 0
 Conv_2_Relu_constraint_1[3,15]: - Conv_2_t[3,15] + Conv_2_Relu_ts[3,15]
   - Conv_2_Relu_ss[3,15] = 0
 Conv_2_Relu_constraint_1[4,0]: - Conv_2_t[4,0] + Conv_2_Relu_ts[4,0]
   - Conv_2_Relu_ss[4,0] = 0
 Conv_2_Relu_constraint_1[4,1]: - Conv_2_t[4,1] + Conv_2_Relu_ts[4,1]
   - Conv_2_Relu_ss[4,1] = 0
 Conv_2_Relu_constraint_1[4,2]: - Conv_2_t[4,2] + Conv_2_Relu_ts[4,2]
   - Conv_2_Relu_ss[4,2] = 0
 Conv_2_Relu_constraint_1[4,3]: - Conv_2_t[4,3] + Conv_2_Relu_ts[4,3]
   - Conv_2_Relu_ss[4,3] = 0
 Conv_2_Relu_constraint_1[4,4]: - Conv_2_t[4,4] + Conv_2_Relu_ts[4,4]
   - Conv_2_Relu_ss[4,4] = 0
 Conv_2_Relu_constraint_1[4,5]: - Conv_2_t[4,5] + Conv_2_Relu_ts[4,5]
   - Conv_2_Relu_ss[4,5] = 0
 Conv_2_Relu_constraint_1[4,6]: - Conv_2_t[4,6] + Conv_2_Relu_ts[4,6]
   - Conv_2_Relu_ss[4,6] = 0
 Conv_2_Relu_constraint_1[4,7]: - Conv_2_t[4,7] + Conv_2_Relu_ts[4,7]
   - Conv_2_Relu_ss[4,7] = 0
 Conv_2_Relu_constraint_1[4,8]: - Conv_2_t[4,8] + Conv_2_Relu_ts[4,8]
   - Conv_2_Relu_ss[4,8] = 0
 Conv_2_Relu_constraint_1[4,9]: - Conv_2_t[4,9] + Conv_2_Relu_ts[4,9]
   - Conv_2_Relu_ss[4,9] = 0
 Conv_2_Relu_constraint_1[4,10]: - Conv_2_t[4,10] + Conv_2_Relu_ts[4,10]
   - Conv_2_Relu_ss[4,10] = 0
 Conv_2_Relu_constraint_1[4,11]: - Conv_2_t[4,11] + Conv_2_Relu_ts[4,11]
   - Conv_2_Relu_ss[4,11] = 0
 Conv_2_Relu_constraint_1[4,12]: - Conv_2_t[4,12] + Conv_2_Relu_ts[4,12]
   - Conv_2_Relu_ss[4,12] = 0
 Conv_2_Relu_constraint_1[4,13]: - Conv_2_t[4,13] + Conv_2_Relu_ts[4,13]
   - Conv_2_Relu_ss[4,13] = 0
 Conv_2_Relu_constraint_1[4,14]: - Conv_2_t[4,14] + Conv_2_Relu_ts[4,14]
   - Conv_2_Relu_ss[4,14] = 0
 Conv_2_Relu_constraint_1[4,15]: - Conv_2_t[4,15] + Conv_2_Relu_ts[4,15]
   - Conv_2_Relu_ss[4,15] = 0
 Conv_2_Relu_constraint_1[5,0]: - Conv_2_t[5,0] + Conv_2_Relu_ts[5,0]
   - Conv_2_Relu_ss[5,0] = 0
 Conv_2_Relu_constraint_1[5,1]: - Conv_2_t[5,1] + Conv_2_Relu_ts[5,1]
   - Conv_2_Relu_ss[5,1] = 0
 Conv_2_Relu_constraint_1[5,2]: - Conv_2_t[5,2] + Conv_2_Relu_ts[5,2]
   - Conv_2_Relu_ss[5,2] = 0
 Conv_2_Relu_constraint_1[5,3]: - Conv_2_t[5,3] + Conv_2_Relu_ts[5,3]
   - Conv_2_Relu_ss[5,3] = 0
 Conv_2_Relu_constraint_1[5,4]: - Conv_2_t[5,4] + Conv_2_Relu_ts[5,4]
   - Conv_2_Relu_ss[5,4] = 0
 Conv_2_Relu_constraint_1[5,5]: - Conv_2_t[5,5] + Conv_2_Relu_ts[5,5]
   - Conv_2_Relu_ss[5,5] = 0
 Conv_2_Relu_constraint_1[5,6]: - Conv_2_t[5,6] + Conv_2_Relu_ts[5,6]
   - Conv_2_Relu_ss[5,6] = 0
 Conv_2_Relu_constraint_1[5,7]: - Conv_2_t[5,7] + Conv_2_Relu_ts[5,7]
   - Conv_2_Relu_ss[5,7] = 0
 Conv_2_Relu_constraint_1[5,8]: - Conv_2_t[5,8] + Conv_2_Relu_ts[5,8]
   - Conv_2_Relu_ss[5,8] = 0
 Conv_2_Relu_constraint_1[5,9]: - Conv_2_t[5,9] + Conv_2_Relu_ts[5,9]
   - Conv_2_Relu_ss[5,9] = 0
 Conv_2_Relu_constraint_1[5,10]: - Conv_2_t[5,10] + Conv_2_Relu_ts[5,10]
   - Conv_2_Relu_ss[5,10] = 0
 Conv_2_Relu_constraint_1[5,11]: - Conv_2_t[5,11] + Conv_2_Relu_ts[5,11]
   - Conv_2_Relu_ss[5,11] = 0
 Conv_2_Relu_constraint_1[5,12]: - Conv_2_t[5,12] + Conv_2_Relu_ts[5,12]
   - Conv_2_Relu_ss[5,12] = 0
 Conv_2_Relu_constraint_1[5,13]: - Conv_2_t[5,13] + Conv_2_Relu_ts[5,13]
   - Conv_2_Relu_ss[5,13] = 0
 Conv_2_Relu_constraint_1[5,14]: - Conv_2_t[5,14] + Conv_2_Relu_ts[5,14]
   - Conv_2_Relu_ss[5,14] = 0
 Conv_2_Relu_constraint_1[5,15]: - Conv_2_t[5,15] + Conv_2_Relu_ts[5,15]
   - Conv_2_Relu_ss[5,15] = 0
 Conv_2_Relu_constraint_1[6,0]: - Conv_2_t[6,0] + Conv_2_Relu_ts[6,0]
   - Conv_2_Relu_ss[6,0] = 0
 Conv_2_Relu_constraint_1[6,1]: - Conv_2_t[6,1] + Conv_2_Relu_ts[6,1]
   - Conv_2_Relu_ss[6,1] = 0
 Conv_2_Relu_constraint_1[6,2]: - Conv_2_t[6,2] + Conv_2_Relu_ts[6,2]
   - Conv_2_Relu_ss[6,2] = 0
 Conv_2_Relu_constraint_1[6,3]: - Conv_2_t[6,3] + Conv_2_Relu_ts[6,3]
   - Conv_2_Relu_ss[6,3] = 0
 Conv_2_Relu_constraint_1[6,4]: - Conv_2_t[6,4] + Conv_2_Relu_ts[6,4]
   - Conv_2_Relu_ss[6,4] = 0
 Conv_2_Relu_constraint_1[6,5]: - Conv_2_t[6,5] + Conv_2_Relu_ts[6,5]
   - Conv_2_Relu_ss[6,5] = 0
 Conv_2_Relu_constraint_1[6,6]: - Conv_2_t[6,6] + Conv_2_Relu_ts[6,6]
   - Conv_2_Relu_ss[6,6] = 0
 Conv_2_Relu_constraint_1[6,7]: - Conv_2_t[6,7] + Conv_2_Relu_ts[6,7]
   - Conv_2_Relu_ss[6,7] = 0
 Conv_2_Relu_constraint_1[6,8]: - Conv_2_t[6,8] + Conv_2_Relu_ts[6,8]
   - Conv_2_Relu_ss[6,8] = 0
 Conv_2_Relu_constraint_1[6,9]: - Conv_2_t[6,9] + Conv_2_Relu_ts[6,9]
   - Conv_2_Relu_ss[6,9] = 0
 Conv_2_Relu_constraint_1[6,10]: - Conv_2_t[6,10] + Conv_2_Relu_ts[6,10]
   - Conv_2_Relu_ss[6,10] = 0
 Conv_2_Relu_constraint_1[6,11]: - Conv_2_t[6,11] + Conv_2_Relu_ts[6,11]
   - Conv_2_Relu_ss[6,11] = 0
 Conv_2_Relu_constraint_1[6,12]: - Conv_2_t[6,12] + Conv_2_Relu_ts[6,12]
   - Conv_2_Relu_ss[6,12] = 0
 Conv_2_Relu_constraint_1[6,13]: - Conv_2_t[6,13] + Conv_2_Relu_ts[6,13]
   - Conv_2_Relu_ss[6,13] = 0
 Conv_2_Relu_constraint_1[6,14]: - Conv_2_t[6,14] + Conv_2_Relu_ts[6,14]
   - Conv_2_Relu_ss[6,14] = 0
 Conv_2_Relu_constraint_1[6,15]: - Conv_2_t[6,15] + Conv_2_Relu_ts[6,15]
   - Conv_2_Relu_ss[6,15] = 0
 Conv_2_Relu_constraint_1[7,0]: - Conv_2_t[7,0] + Conv_2_Relu_ts[7,0]
   - Conv_2_Relu_ss[7,0] = 0
 Conv_2_Relu_constraint_1[7,1]: - Conv_2_t[7,1] + Conv_2_Relu_ts[7,1]
   - Conv_2_Relu_ss[7,1] = 0
 Conv_2_Relu_constraint_1[7,2]: - Conv_2_t[7,2] + Conv_2_Relu_ts[7,2]
   - Conv_2_Relu_ss[7,2] = 0
 Conv_2_Relu_constraint_1[7,3]: - Conv_2_t[7,3] + Conv_2_Relu_ts[7,3]
   - Conv_2_Relu_ss[7,3] = 0
 Conv_2_Relu_constraint_1[7,4]: - Conv_2_t[7,4] + Conv_2_Relu_ts[7,4]
   - Conv_2_Relu_ss[7,4] = 0
 Conv_2_Relu_constraint_1[7,5]: - Conv_2_t[7,5] + Conv_2_Relu_ts[7,5]
   - Conv_2_Relu_ss[7,5] = 0
 Conv_2_Relu_constraint_1[7,6]: - Conv_2_t[7,6] + Conv_2_Relu_ts[7,6]
   - Conv_2_Relu_ss[7,6] = 0
 Conv_2_Relu_constraint_1[7,7]: - Conv_2_t[7,7] + Conv_2_Relu_ts[7,7]
   - Conv_2_Relu_ss[7,7] = 0
 Conv_2_Relu_constraint_1[7,8]: - Conv_2_t[7,8] + Conv_2_Relu_ts[7,8]
   - Conv_2_Relu_ss[7,8] = 0
 Conv_2_Relu_constraint_1[7,9]: - Conv_2_t[7,9] + Conv_2_Relu_ts[7,9]
   - Conv_2_Relu_ss[7,9] = 0
 Conv_2_Relu_constraint_1[7,10]: - Conv_2_t[7,10] + Conv_2_Relu_ts[7,10]
   - Conv_2_Relu_ss[7,10] = 0
 Conv_2_Relu_constraint_1[7,11]: - Conv_2_t[7,11] + Conv_2_Relu_ts[7,11]
   - Conv_2_Relu_ss[7,11] = 0
 Conv_2_Relu_constraint_1[7,12]: - Conv_2_t[7,12] + Conv_2_Relu_ts[7,12]
   - Conv_2_Relu_ss[7,12] = 0
 Conv_2_Relu_constraint_1[7,13]: - Conv_2_t[7,13] + Conv_2_Relu_ts[7,13]
   - Conv_2_Relu_ss[7,13] = 0
 Conv_2_Relu_constraint_1[7,14]: - Conv_2_t[7,14] + Conv_2_Relu_ts[7,14]
   - Conv_2_Relu_ss[7,14] = 0
 Conv_2_Relu_constraint_1[7,15]: - Conv_2_t[7,15] + Conv_2_Relu_ts[7,15]
   - Conv_2_Relu_ss[7,15] = 0
 Conv_2_Relu_constraint_2[0,0]: Conv_2_Relu_ts[0,0] - 5.88025e-13 C1744
   <= 0
 Conv_2_Relu_constraint_2[0,1]: Conv_2_Relu_ts[0,1]
   - 4.5736554106570173e-04 C1745 <= 0
 Conv_2_Relu_constraint_2[0,2]: Conv_2_Relu_ts[0,2] <= 0
 Conv_2_Relu_constraint_2[0,3]: Conv_2_Relu_ts[0,3] - 4.99422e-05 C1747
   <= 0
 Conv_2_Relu_constraint_2[0,4]: Conv_2_Relu_ts[0,4] - 1.48705e-05 C1748
   <= 0
 Conv_2_Relu_constraint_2[0,5]: Conv_2_Relu_ts[0,5] - 6.86581e-09 C1749
   <= 0
 Conv_2_Relu_constraint_2[0,6]: Conv_2_Relu_ts[0,6]
   - 13.42537958453236 C1750 <= 0
 Conv_2_Relu_constraint_2[0,7]: Conv_2_Relu_ts[0,7]
   - 11.98662083090122 C1751 <= 0
 Conv_2_Relu_constraint_2[0,8]: Conv_2_Relu_ts[0,8] - 7.6174e-05 C1752
   <= 0
 Conv_2_Relu_constraint_2[0,9]: Conv_2_Relu_ts[0,9]
   - 0.00112970281782641 C1753 <= 0
 Conv_2_Relu_constraint_2[0,10]: Conv_2_Relu_ts[0,10] <= 0
 Conv_2_Relu_constraint_2[0,11]: Conv_2_Relu_ts[0,11] - 5.14323e-07 C1755
   <= 0
 Conv_2_Relu_constraint_2[0,12]: Conv_2_Relu_ts[0,12]
   - 19.55128642019131 C1756 <= 0
 Conv_2_Relu_constraint_2[0,13]: Conv_2_Relu_ts[0,13]
   - 0.8449266835182555 C1757 <= 0
 Conv_2_Relu_constraint_2[0,14]: Conv_2_Relu_ts[0,14]
   - 17.02833982748556 C1758 <= 0
 Conv_2_Relu_constraint_2[0,15]: Conv_2_Relu_ts[0,15]
   - 1.8051988252351864e-04 C1759 <= 0
 Conv_2_Relu_constraint_2[1,0]: Conv_2_Relu_ts[1,0] - 5.88025e-13 C1760
   <= 0
 Conv_2_Relu_constraint_2[1,1]: Conv_2_Relu_ts[1,1]
   - 4.5736554106570173e-04 C1761 <= 0
 Conv_2_Relu_constraint_2[1,2]: Conv_2_Relu_ts[1,2] <= 0
 Conv_2_Relu_constraint_2[1,3]: Conv_2_Relu_ts[1,3] - 4.99422e-05 C1763
   <= 0
 Conv_2_Relu_constraint_2[1,4]: Conv_2_Relu_ts[1,4] - 1.48705e-05 C1764
   <= 0
 Conv_2_Relu_constraint_2[1,5]: Conv_2_Relu_ts[1,5] - 6.86581e-09 C1765
   <= 0
 Conv_2_Relu_constraint_2[1,6]: Conv_2_Relu_ts[1,6]
   - 13.42537958453236 C1766 <= 0
 Conv_2_Relu_constraint_2[1,7]: Conv_2_Relu_ts[1,7]
   - 11.98662083090122 C1767 <= 0
 Conv_2_Relu_constraint_2[1,8]: Conv_2_Relu_ts[1,8] - 7.6174e-05 C1768
   <= 0
 Conv_2_Relu_constraint_2[1,9]: Conv_2_Relu_ts[1,9]
   - 0.00112970281782641 C1769 <= 0
 Conv_2_Relu_constraint_2[1,10]: Conv_2_Relu_ts[1,10] <= 0
 Conv_2_Relu_constraint_2[1,11]: Conv_2_Relu_ts[1,11] - 5.14323e-07 C1771
   <= 0
 Conv_2_Relu_constraint_2[1,12]: Conv_2_Relu_ts[1,12]
   - 19.55128642019131 C1772 <= 0
 Conv_2_Relu_constraint_2[1,13]: Conv_2_Relu_ts[1,13]
   - 0.8449266835182555 C1773 <= 0
 Conv_2_Relu_constraint_2[1,14]: Conv_2_Relu_ts[1,14]
   - 17.02833982748556 C1774 <= 0
 Conv_2_Relu_constraint_2[1,15]: Conv_2_Relu_ts[1,15]
   - 1.8051988252351864e-04 C1775 <= 0
 Conv_2_Relu_constraint_2[2,0]: Conv_2_Relu_ts[2,0] - 5.88025e-13 C1776
   <= 0
 Conv_2_Relu_constraint_2[2,1]: Conv_2_Relu_ts[2,1]
   - 4.5736554106570173e-04 C1777 <= 0
 Conv_2_Relu_constraint_2[2,2]: Conv_2_Relu_ts[2,2] <= 0
 Conv_2_Relu_constraint_2[2,3]: Conv_2_Relu_ts[2,3] - 4.99422e-05 C1779
   <= 0
 Conv_2_Relu_constraint_2[2,4]: Conv_2_Relu_ts[2,4] - 1.48705e-05 C1780
   <= 0
 Conv_2_Relu_constraint_2[2,5]: Conv_2_Relu_ts[2,5] - 6.86581e-09 C1781
   <= 0
 Conv_2_Relu_constraint_2[2,6]: Conv_2_Relu_ts[2,6]
   - 13.42537958453236 C1782 <= 0
 Conv_2_Relu_constraint_2[2,7]: Conv_2_Relu_ts[2,7]
   - 11.98662083090122 C1783 <= 0
 Conv_2_Relu_constraint_2[2,8]: Conv_2_Relu_ts[2,8] - 7.6174e-05 C1784
   <= 0
 Conv_2_Relu_constraint_2[2,9]: Conv_2_Relu_ts[2,9]
   - 0.00112970281782641 C1785 <= 0
 Conv_2_Relu_constraint_2[2,10]: Conv_2_Relu_ts[2,10] <= 0
 Conv_2_Relu_constraint_2[2,11]: Conv_2_Relu_ts[2,11] - 5.14323e-07 C1787
   <= 0
 Conv_2_Relu_constraint_2[2,12]: Conv_2_Relu_ts[2,12]
   - 19.55128642019131 C1788 <= 0
 Conv_2_Relu_constraint_2[2,13]: Conv_2_Relu_ts[2,13]
   - 0.8449266835182555 C1789 <= 0
 Conv_2_Relu_constraint_2[2,14]: Conv_2_Relu_ts[2,14]
   - 17.02833982748556 C1790 <= 0
 Conv_2_Relu_constraint_2[2,15]: Conv_2_Relu_ts[2,15]
   - 1.8051988252351864e-04 C1791 <= 0
 Conv_2_Relu_constraint_2[3,0]: Conv_2_Relu_ts[3,0] - 5.88025e-13 C1792
   <= 0
 Conv_2_Relu_constraint_2[3,1]: Conv_2_Relu_ts[3,1]
   - 4.5736554106570173e-04 C1793 <= 0
 Conv_2_Relu_constraint_2[3,2]: Conv_2_Relu_ts[3,2] <= 0
 Conv_2_Relu_constraint_2[3,3]: Conv_2_Relu_ts[3,3] - 4.99422e-05 C1795
   <= 0
 Conv_2_Relu_constraint_2[3,4]: Conv_2_Relu_ts[3,4] - 1.48705e-05 C1796
   <= 0
 Conv_2_Relu_constraint_2[3,5]: Conv_2_Relu_ts[3,5] - 6.86581e-09 C1797
   <= 0
 Conv_2_Relu_constraint_2[3,6]: Conv_2_Relu_ts[3,6]
   - 13.42537958453236 C1798 <= 0
 Conv_2_Relu_constraint_2[3,7]: Conv_2_Relu_ts[3,7]
   - 11.98662083090122 C1799 <= 0
 Conv_2_Relu_constraint_2[3,8]: Conv_2_Relu_ts[3,8] - 7.6174e-05 C1800
   <= 0
 Conv_2_Relu_constraint_2[3,9]: Conv_2_Relu_ts[3,9]
   - 0.00112970281782641 C1801 <= 0
 Conv_2_Relu_constraint_2[3,10]: Conv_2_Relu_ts[3,10] <= 0
 Conv_2_Relu_constraint_2[3,11]: Conv_2_Relu_ts[3,11] - 5.14323e-07 C1803
   <= 0
 Conv_2_Relu_constraint_2[3,12]: Conv_2_Relu_ts[3,12]
   - 19.55128642019131 C1804 <= 0
 Conv_2_Relu_constraint_2[3,13]: Conv_2_Relu_ts[3,13]
   - 0.8449266835182555 C1805 <= 0
 Conv_2_Relu_constraint_2[3,14]: Conv_2_Relu_ts[3,14]
   - 17.02833982748556 C1806 <= 0
 Conv_2_Relu_constraint_2[3,15]: Conv_2_Relu_ts[3,15]
   - 1.8051988252351864e-04 C1807 <= 0
 Conv_2_Relu_constraint_2[4,0]: Conv_2_Relu_ts[4,0] - 5.88025e-13 C1808
   <= 0
 Conv_2_Relu_constraint_2[4,1]: Conv_2_Relu_ts[4,1]
   - 4.5736554106570173e-04 C1809 <= 0
 Conv_2_Relu_constraint_2[4,2]: Conv_2_Relu_ts[4,2] <= 0
 Conv_2_Relu_constraint_2[4,3]: Conv_2_Relu_ts[4,3] - 4.99422e-05 C1811
   <= 0
 Conv_2_Relu_constraint_2[4,4]: Conv_2_Relu_ts[4,4] - 1.48705e-05 C1812
   <= 0
 Conv_2_Relu_constraint_2[4,5]: Conv_2_Relu_ts[4,5] - 6.86581e-09 C1813
   <= 0
 Conv_2_Relu_constraint_2[4,6]: Conv_2_Relu_ts[4,6]
   - 13.42537958453236 C1814 <= 0
 Conv_2_Relu_constraint_2[4,7]: Conv_2_Relu_ts[4,7]
   - 11.98662083090122 C1815 <= 0
 Conv_2_Relu_constraint_2[4,8]: Conv_2_Relu_ts[4,8] - 7.6174e-05 C1816
   <= 0
 Conv_2_Relu_constraint_2[4,9]: Conv_2_Relu_ts[4,9]
   - 0.00112970281782641 C1817 <= 0
 Conv_2_Relu_constraint_2[4,10]: Conv_2_Relu_ts[4,10] <= 0
 Conv_2_Relu_constraint_2[4,11]: Conv_2_Relu_ts[4,11] - 5.14323e-07 C1819
   <= 0
 Conv_2_Relu_constraint_2[4,12]: Conv_2_Relu_ts[4,12]
   - 19.55128642019131 C1820 <= 0
 Conv_2_Relu_constraint_2[4,13]: Conv_2_Relu_ts[4,13]
   - 0.8449266835182555 C1821 <= 0
 Conv_2_Relu_constraint_2[4,14]: Conv_2_Relu_ts[4,14]
   - 17.02833982748556 C1822 <= 0
 Conv_2_Relu_constraint_2[4,15]: Conv_2_Relu_ts[4,15]
   - 1.8051988252351864e-04 C1823 <= 0
 Conv_2_Relu_constraint_2[5,0]: Conv_2_Relu_ts[5,0] - 5.88025e-13 C1824
   <= 0
 Conv_2_Relu_constraint_2[5,1]: Conv_2_Relu_ts[5,1]
   - 4.5736554106570173e-04 C1825 <= 0
 Conv_2_Relu_constraint_2[5,2]: Conv_2_Relu_ts[5,2] <= 0
 Conv_2_Relu_constraint_2[5,3]: Conv_2_Relu_ts[5,3] - 4.99422e-05 C1827
   <= 0
 Conv_2_Relu_constraint_2[5,4]: Conv_2_Relu_ts[5,4] - 1.48705e-05 C1828
   <= 0
 Conv_2_Relu_constraint_2[5,5]: Conv_2_Relu_ts[5,5] - 6.86581e-09 C1829
   <= 0
 Conv_2_Relu_constraint_2[5,6]: Conv_2_Relu_ts[5,6]
   - 13.42537958453236 C1830 <= 0
 Conv_2_Relu_constraint_2[5,7]: Conv_2_Relu_ts[5,7]
   - 11.98662083090122 C1831 <= 0
 Conv_2_Relu_constraint_2[5,8]: Conv_2_Relu_ts[5,8] - 7.6174e-05 C1832
   <= 0
 Conv_2_Relu_constraint_2[5,9]: Conv_2_Relu_ts[5,9]
   - 0.00112970281782641 C1833 <= 0
 Conv_2_Relu_constraint_2[5,10]: Conv_2_Relu_ts[5,10] <= 0
 Conv_2_Relu_constraint_2[5,11]: Conv_2_Relu_ts[5,11] - 5.14323e-07 C1835
   <= 0
 Conv_2_Relu_constraint_2[5,12]: Conv_2_Relu_ts[5,12]
   - 19.55128642019131 C1836 <= 0
 Conv_2_Relu_constraint_2[5,13]: Conv_2_Relu_ts[5,13]
   - 0.8449266835182555 C1837 <= 0
 Conv_2_Relu_constraint_2[5,14]: Conv_2_Relu_ts[5,14]
   - 17.02833982748556 C1838 <= 0
 Conv_2_Relu_constraint_2[5,15]: Conv_2_Relu_ts[5,15]
   - 1.8051988252351864e-04 C1839 <= 0
 Conv_2_Relu_constraint_2[6,0]: Conv_2_Relu_ts[6,0] - 5.88025e-13 C1840
   <= 0
 Conv_2_Relu_constraint_2[6,1]: Conv_2_Relu_ts[6,1]
   - 4.5736554106570173e-04 C1841 <= 0
 Conv_2_Relu_constraint_2[6,2]: Conv_2_Relu_ts[6,2] <= 0
 Conv_2_Relu_constraint_2[6,3]: Conv_2_Relu_ts[6,3] - 4.99422e-05 C1843
   <= 0
 Conv_2_Relu_constraint_2[6,4]: Conv_2_Relu_ts[6,4] - 1.48705e-05 C1844
   <= 0
 Conv_2_Relu_constraint_2[6,5]: Conv_2_Relu_ts[6,5] - 6.86581e-09 C1845
   <= 0
 Conv_2_Relu_constraint_2[6,6]: Conv_2_Relu_ts[6,6]
   - 13.42537958453236 C1846 <= 0
 Conv_2_Relu_constraint_2[6,7]: Conv_2_Relu_ts[6,7]
   - 11.98662083090122 C1847 <= 0
 Conv_2_Relu_constraint_2[6,8]: Conv_2_Relu_ts[6,8] - 7.6174e-05 C1848
   <= 0
 Conv_2_Relu_constraint_2[6,9]: Conv_2_Relu_ts[6,9]
   - 0.00112970281782641 C1849 <= 0
 Conv_2_Relu_constraint_2[6,10]: Conv_2_Relu_ts[6,10] <= 0
 Conv_2_Relu_constraint_2[6,11]: Conv_2_Relu_ts[6,11] - 5.14323e-07 C1851
   <= 0
 Conv_2_Relu_constraint_2[6,12]: Conv_2_Relu_ts[6,12]
   - 19.55128642019131 C1852 <= 0
 Conv_2_Relu_constraint_2[6,13]: Conv_2_Relu_ts[6,13]
   - 0.8449266835182555 C1853 <= 0
 Conv_2_Relu_constraint_2[6,14]: Conv_2_Relu_ts[6,14]
   - 17.02833982748556 C1854 <= 0
 Conv_2_Relu_constraint_2[6,15]: Conv_2_Relu_ts[6,15]
   - 1.8051988252351864e-04 C1855 <= 0
 Conv_2_Relu_constraint_2[7,0]: Conv_2_Relu_ts[7,0] - 5.88025e-13 C1856
   <= 0
 Conv_2_Relu_constraint_2[7,1]: Conv_2_Relu_ts[7,1]
   - 4.5736554106570173e-04 C1857 <= 0
 Conv_2_Relu_constraint_2[7,2]: Conv_2_Relu_ts[7,2] <= 0
 Conv_2_Relu_constraint_2[7,3]: Conv_2_Relu_ts[7,3] - 4.99422e-05 C1859
   <= 0
 Conv_2_Relu_constraint_2[7,4]: Conv_2_Relu_ts[7,4] - 1.48705e-05 C1860
   <= 0
 Conv_2_Relu_constraint_2[7,5]: Conv_2_Relu_ts[7,5] - 6.86581e-09 C1861
   <= 0
 Conv_2_Relu_constraint_2[7,6]: Conv_2_Relu_ts[7,6]
   - 13.42537958453236 C1862 <= 0
 Conv_2_Relu_constraint_2[7,7]: Conv_2_Relu_ts[7,7]
   - 11.98662083090122 C1863 <= 0
 Conv_2_Relu_constraint_2[7,8]: Conv_2_Relu_ts[7,8] - 7.6174e-05 C1864
   <= 0
 Conv_2_Relu_constraint_2[7,9]: Conv_2_Relu_ts[7,9]
   - 0.00112970281782641 C1865 <= 0
 Conv_2_Relu_constraint_2[7,10]: Conv_2_Relu_ts[7,10] <= 0
 Conv_2_Relu_constraint_2[7,11]: Conv_2_Relu_ts[7,11] - 5.14323e-07 C1867
   <= 0
 Conv_2_Relu_constraint_2[7,12]: Conv_2_Relu_ts[7,12]
   - 19.55128642019131 C1868 <= 0
 Conv_2_Relu_constraint_2[7,13]: Conv_2_Relu_ts[7,13]
   - 0.8449266835182555 C1869 <= 0
 Conv_2_Relu_constraint_2[7,14]: Conv_2_Relu_ts[7,14]
   - 17.02833982748556 C1870 <= 0
 Conv_2_Relu_constraint_2[7,15]: Conv_2_Relu_ts[7,15]
   - 1.8051988252351864e-04 C1871 <= 0
 Conv_2_Relu_constraint_3[0,0]: Conv_2_Relu_ss[0,0] + 2.60542e-12 C1744
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[0,1]: Conv_2_Relu_ss[0,1]
   + 0.00259094797606434 C1745 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[0,2]: Conv_2_Relu_ss[0,2] + 3.60829e-10 C1746
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[0,3]: Conv_2_Relu_ss[0,3] <= 0
 Conv_2_Relu_constraint_3[0,4]: Conv_2_Relu_ss[0,4]
   + 0.0176422467492259 C1748 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[0,5]: Conv_2_Relu_ss[0,5] + 2.51833e-08 C1749
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[0,6]: Conv_2_Relu_ss[0,6]
   + 38.08457690166041 C1750 <= 38.08457690166041
 Conv_2_Relu_constraint_3[0,7]: Conv_2_Relu_ss[0,7]
   + 6.983479580766037 C1751 <= 6.983479580766037
 Conv_2_Relu_constraint_3[0,8]: Conv_2_Relu_ss[0,8]
   + 1.9980672280076582e-04 C1752 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[0,9]: Conv_2_Relu_ss[0,9] <= 0
 Conv_2_Relu_constraint_3[0,10]: Conv_2_Relu_ss[0,10] + 4.8062e-07 C1754
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[0,11]: Conv_2_Relu_ss[0,11] <= 0
 Conv_2_Relu_constraint_3[0,12]: Conv_2_Relu_ss[0,12]
   + 6.67338989035037 C1756 <= 6.67338989035037
 Conv_2_Relu_constraint_3[0,13]: Conv_2_Relu_ss[0,13]
   + 0.7895277949390602 C1757 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[0,14]: Conv_2_Relu_ss[0,14] <= 0
 Conv_2_Relu_constraint_3[0,15]: Conv_2_Relu_ss[0,15] + 0.000225001 C1759
   <= 0.000225001
 Conv_2_Relu_constraint_3[1,0]: Conv_2_Relu_ss[1,0] + 2.60542e-12 C1760
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[1,1]: Conv_2_Relu_ss[1,1]
   + 0.00259094797606434 C1761 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[1,2]: Conv_2_Relu_ss[1,2] + 3.60829e-10 C1762
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[1,3]: Conv_2_Relu_ss[1,3] <= 0
 Conv_2_Relu_constraint_3[1,4]: Conv_2_Relu_ss[1,4]
   + 0.0176422467492259 C1764 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[1,5]: Conv_2_Relu_ss[1,5] + 2.51833e-08 C1765
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[1,6]: Conv_2_Relu_ss[1,6]
   + 38.08457690166041 C1766 <= 38.08457690166041
 Conv_2_Relu_constraint_3[1,7]: Conv_2_Relu_ss[1,7]
   + 6.983479580766037 C1767 <= 6.983479580766037
 Conv_2_Relu_constraint_3[1,8]: Conv_2_Relu_ss[1,8]
   + 1.9980672280076582e-04 C1768 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[1,9]: Conv_2_Relu_ss[1,9] <= 0
 Conv_2_Relu_constraint_3[1,10]: Conv_2_Relu_ss[1,10] + 4.8062e-07 C1770
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[1,11]: Conv_2_Relu_ss[1,11] <= 0
 Conv_2_Relu_constraint_3[1,12]: Conv_2_Relu_ss[1,12]
   + 6.67338989035037 C1772 <= 6.67338989035037
 Conv_2_Relu_constraint_3[1,13]: Conv_2_Relu_ss[1,13]
   + 0.7895277949390602 C1773 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[1,14]: Conv_2_Relu_ss[1,14] <= 0
 Conv_2_Relu_constraint_3[1,15]: Conv_2_Relu_ss[1,15] + 0.000225001 C1775
   <= 0.000225001
 Conv_2_Relu_constraint_3[2,0]: Conv_2_Relu_ss[2,0] + 2.60542e-12 C1776
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[2,1]: Conv_2_Relu_ss[2,1]
   + 0.00259094797606434 C1777 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[2,2]: Conv_2_Relu_ss[2,2] + 3.60829e-10 C1778
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[2,3]: Conv_2_Relu_ss[2,3] <= 0
 Conv_2_Relu_constraint_3[2,4]: Conv_2_Relu_ss[2,4]
   + 0.0176422467492259 C1780 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[2,5]: Conv_2_Relu_ss[2,5] + 2.51833e-08 C1781
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[2,6]: Conv_2_Relu_ss[2,6]
   + 38.08457690166041 C1782 <= 38.08457690166041
 Conv_2_Relu_constraint_3[2,7]: Conv_2_Relu_ss[2,7]
   + 6.983479580766037 C1783 <= 6.983479580766037
 Conv_2_Relu_constraint_3[2,8]: Conv_2_Relu_ss[2,8]
   + 1.9980672280076582e-04 C1784 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[2,9]: Conv_2_Relu_ss[2,9] <= 0
 Conv_2_Relu_constraint_3[2,10]: Conv_2_Relu_ss[2,10] + 4.8062e-07 C1786
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[2,11]: Conv_2_Relu_ss[2,11] <= 0
 Conv_2_Relu_constraint_3[2,12]: Conv_2_Relu_ss[2,12]
   + 6.67338989035037 C1788 <= 6.67338989035037
 Conv_2_Relu_constraint_3[2,13]: Conv_2_Relu_ss[2,13]
   + 0.7895277949390602 C1789 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[2,14]: Conv_2_Relu_ss[2,14] <= 0
 Conv_2_Relu_constraint_3[2,15]: Conv_2_Relu_ss[2,15] + 0.000225001 C1791
   <= 0.000225001
 Conv_2_Relu_constraint_3[3,0]: Conv_2_Relu_ss[3,0] + 2.60542e-12 C1792
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[3,1]: Conv_2_Relu_ss[3,1]
   + 0.00259094797606434 C1793 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[3,2]: Conv_2_Relu_ss[3,2] + 3.60829e-10 C1794
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[3,3]: Conv_2_Relu_ss[3,3] <= 0
 Conv_2_Relu_constraint_3[3,4]: Conv_2_Relu_ss[3,4]
   + 0.0176422467492259 C1796 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[3,5]: Conv_2_Relu_ss[3,5] + 2.51833e-08 C1797
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[3,6]: Conv_2_Relu_ss[3,6]
   + 38.08457690166041 C1798 <= 38.08457690166041
 Conv_2_Relu_constraint_3[3,7]: Conv_2_Relu_ss[3,7]
   + 6.983479580766037 C1799 <= 6.983479580766037
 Conv_2_Relu_constraint_3[3,8]: Conv_2_Relu_ss[3,8]
   + 1.9980672280076582e-04 C1800 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[3,9]: Conv_2_Relu_ss[3,9] <= 0
 Conv_2_Relu_constraint_3[3,10]: Conv_2_Relu_ss[3,10] + 4.8062e-07 C1802
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[3,11]: Conv_2_Relu_ss[3,11] <= 0
 Conv_2_Relu_constraint_3[3,12]: Conv_2_Relu_ss[3,12]
   + 6.67338989035037 C1804 <= 6.67338989035037
 Conv_2_Relu_constraint_3[3,13]: Conv_2_Relu_ss[3,13]
   + 0.7895277949390602 C1805 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[3,14]: Conv_2_Relu_ss[3,14] <= 0
 Conv_2_Relu_constraint_3[3,15]: Conv_2_Relu_ss[3,15] + 0.000225001 C1807
   <= 0.000225001
 Conv_2_Relu_constraint_3[4,0]: Conv_2_Relu_ss[4,0] + 2.60542e-12 C1808
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[4,1]: Conv_2_Relu_ss[4,1]
   + 0.00259094797606434 C1809 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[4,2]: Conv_2_Relu_ss[4,2] + 3.60829e-10 C1810
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[4,3]: Conv_2_Relu_ss[4,3] <= 0
 Conv_2_Relu_constraint_3[4,4]: Conv_2_Relu_ss[4,4]
   + 0.0176422467492259 C1812 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[4,5]: Conv_2_Relu_ss[4,5] + 2.51833e-08 C1813
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[4,6]: Conv_2_Relu_ss[4,6]
   + 38.08457690166041 C1814 <= 38.08457690166041
 Conv_2_Relu_constraint_3[4,7]: Conv_2_Relu_ss[4,7]
   + 6.983479580766037 C1815 <= 6.983479580766037
 Conv_2_Relu_constraint_3[4,8]: Conv_2_Relu_ss[4,8]
   + 1.9980672280076582e-04 C1816 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[4,9]: Conv_2_Relu_ss[4,9] <= 0
 Conv_2_Relu_constraint_3[4,10]: Conv_2_Relu_ss[4,10] + 4.8062e-07 C1818
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[4,11]: Conv_2_Relu_ss[4,11] <= 0
 Conv_2_Relu_constraint_3[4,12]: Conv_2_Relu_ss[4,12]
   + 6.67338989035037 C1820 <= 6.67338989035037
 Conv_2_Relu_constraint_3[4,13]: Conv_2_Relu_ss[4,13]
   + 0.7895277949390602 C1821 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[4,14]: Conv_2_Relu_ss[4,14] <= 0
 Conv_2_Relu_constraint_3[4,15]: Conv_2_Relu_ss[4,15] + 0.000225001 C1823
   <= 0.000225001
 Conv_2_Relu_constraint_3[5,0]: Conv_2_Relu_ss[5,0] + 2.60542e-12 C1824
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[5,1]: Conv_2_Relu_ss[5,1]
   + 0.00259094797606434 C1825 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[5,2]: Conv_2_Relu_ss[5,2] + 3.60829e-10 C1826
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[5,3]: Conv_2_Relu_ss[5,3] <= 0
 Conv_2_Relu_constraint_3[5,4]: Conv_2_Relu_ss[5,4]
   + 0.0176422467492259 C1828 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[5,5]: Conv_2_Relu_ss[5,5] + 2.51833e-08 C1829
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[5,6]: Conv_2_Relu_ss[5,6]
   + 38.08457690166041 C1830 <= 38.08457690166041
 Conv_2_Relu_constraint_3[5,7]: Conv_2_Relu_ss[5,7]
   + 6.983479580766037 C1831 <= 6.983479580766037
 Conv_2_Relu_constraint_3[5,8]: Conv_2_Relu_ss[5,8]
   + 1.9980672280076582e-04 C1832 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[5,9]: Conv_2_Relu_ss[5,9] <= 0
 Conv_2_Relu_constraint_3[5,10]: Conv_2_Relu_ss[5,10] + 4.8062e-07 C1834
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[5,11]: Conv_2_Relu_ss[5,11] <= 0
 Conv_2_Relu_constraint_3[5,12]: Conv_2_Relu_ss[5,12]
   + 6.67338989035037 C1836 <= 6.67338989035037
 Conv_2_Relu_constraint_3[5,13]: Conv_2_Relu_ss[5,13]
   + 0.7895277949390602 C1837 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[5,14]: Conv_2_Relu_ss[5,14] <= 0
 Conv_2_Relu_constraint_3[5,15]: Conv_2_Relu_ss[5,15] + 0.000225001 C1839
   <= 0.000225001
 Conv_2_Relu_constraint_3[6,0]: Conv_2_Relu_ss[6,0] + 2.60542e-12 C1840
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[6,1]: Conv_2_Relu_ss[6,1]
   + 0.00259094797606434 C1841 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[6,2]: Conv_2_Relu_ss[6,2] + 3.60829e-10 C1842
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[6,3]: Conv_2_Relu_ss[6,3] <= 0
 Conv_2_Relu_constraint_3[6,4]: Conv_2_Relu_ss[6,4]
   + 0.0176422467492259 C1844 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[6,5]: Conv_2_Relu_ss[6,5] + 2.51833e-08 C1845
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[6,6]: Conv_2_Relu_ss[6,6]
   + 38.08457690166041 C1846 <= 38.08457690166041
 Conv_2_Relu_constraint_3[6,7]: Conv_2_Relu_ss[6,7]
   + 6.983479580766037 C1847 <= 6.983479580766037
 Conv_2_Relu_constraint_3[6,8]: Conv_2_Relu_ss[6,8]
   + 1.9980672280076582e-04 C1848 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[6,9]: Conv_2_Relu_ss[6,9] <= 0
 Conv_2_Relu_constraint_3[6,10]: Conv_2_Relu_ss[6,10] + 4.8062e-07 C1850
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[6,11]: Conv_2_Relu_ss[6,11] <= 0
 Conv_2_Relu_constraint_3[6,12]: Conv_2_Relu_ss[6,12]
   + 6.67338989035037 C1852 <= 6.67338989035037
 Conv_2_Relu_constraint_3[6,13]: Conv_2_Relu_ss[6,13]
   + 0.7895277949390602 C1853 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[6,14]: Conv_2_Relu_ss[6,14] <= 0
 Conv_2_Relu_constraint_3[6,15]: Conv_2_Relu_ss[6,15] + 0.000225001 C1855
   <= 0.000225001
 Conv_2_Relu_constraint_3[7,0]: Conv_2_Relu_ss[7,0] + 2.60542e-12 C1856
   <= 2.60542e-12
 Conv_2_Relu_constraint_3[7,1]: Conv_2_Relu_ss[7,1]
   + 0.00259094797606434 C1857 <= 0.00259094797606434
 Conv_2_Relu_constraint_3[7,2]: Conv_2_Relu_ss[7,2] + 3.60829e-10 C1858
   <= 3.60829e-10
 Conv_2_Relu_constraint_3[7,3]: Conv_2_Relu_ss[7,3] <= 0
 Conv_2_Relu_constraint_3[7,4]: Conv_2_Relu_ss[7,4]
   + 0.0176422467492259 C1860 <= 0.0176422467492259
 Conv_2_Relu_constraint_3[7,5]: Conv_2_Relu_ss[7,5] + 2.51833e-08 C1861
   <= 2.51833e-08
 Conv_2_Relu_constraint_3[7,6]: Conv_2_Relu_ss[7,6]
   + 38.08457690166041 C1862 <= 38.08457690166041
 Conv_2_Relu_constraint_3[7,7]: Conv_2_Relu_ss[7,7]
   + 6.983479580766037 C1863 <= 6.983479580766037
 Conv_2_Relu_constraint_3[7,8]: Conv_2_Relu_ss[7,8]
   + 1.9980672280076582e-04 C1864 <= 1.9980672280076582e-04
 Conv_2_Relu_constraint_3[7,9]: Conv_2_Relu_ss[7,9] <= 0
 Conv_2_Relu_constraint_3[7,10]: Conv_2_Relu_ss[7,10] + 4.8062e-07 C1866
   <= 4.8062e-07
 Conv_2_Relu_constraint_3[7,11]: Conv_2_Relu_ss[7,11] <= 0
 Conv_2_Relu_constraint_3[7,12]: Conv_2_Relu_ss[7,12]
   + 6.67338989035037 C1868 <= 6.67338989035037
 Conv_2_Relu_constraint_3[7,13]: Conv_2_Relu_ss[7,13]
   + 0.7895277949390602 C1869 <= 0.7895277949390602
 Conv_2_Relu_constraint_3[7,14]: Conv_2_Relu_ss[7,14] <= 0
 Conv_2_Relu_constraint_3[7,15]: Conv_2_Relu_ss[7,15] + 0.000225001 C1871
   <= 0.000225001
 Aggregation_constraint[0]: - 0.125 Conv_2_Relu_ts[0,0]
   - 0.125 Conv_2_Relu_ts[1,0] - 0.125 Conv_2_Relu_ts[2,0]
   - 0.125 Conv_2_Relu_ts[3,0] - 0.125 Conv_2_Relu_ts[4,0]
   - 0.125 Conv_2_Relu_ts[5,0] - 0.125 Conv_2_Relu_ts[6,0]
   - 0.125 Conv_2_Relu_ts[7,0] + Aggregation[0] = 0
 Aggregation_constraint[1]: - 0.125 Conv_2_Relu_ts[0,1]
   - 0.125 Conv_2_Relu_ts[1,1] - 0.125 Conv_2_Relu_ts[2,1]
   - 0.125 Conv_2_Relu_ts[3,1] - 0.125 Conv_2_Relu_ts[4,1]
   - 0.125 Conv_2_Relu_ts[5,1] - 0.125 Conv_2_Relu_ts[6,1]
   - 0.125 Conv_2_Relu_ts[7,1] + Aggregation[1] = 0
 Aggregation_constraint[2]: - 0.125 Conv_2_Relu_ts[0,2]
   - 0.125 Conv_2_Relu_ts[1,2] - 0.125 Conv_2_Relu_ts[2,2]
   - 0.125 Conv_2_Relu_ts[3,2] - 0.125 Conv_2_Relu_ts[4,2]
   - 0.125 Conv_2_Relu_ts[5,2] - 0.125 Conv_2_Relu_ts[6,2]
   - 0.125 Conv_2_Relu_ts[7,2] + Aggregation[2] = 0
 Aggregation_constraint[3]: - 0.125 Conv_2_Relu_ts[0,3]
   - 0.125 Conv_2_Relu_ts[1,3] - 0.125 Conv_2_Relu_ts[2,3]
   - 0.125 Conv_2_Relu_ts[3,3] - 0.125 Conv_2_Relu_ts[4,3]
   - 0.125 Conv_2_Relu_ts[5,3] - 0.125 Conv_2_Relu_ts[6,3]
   - 0.125 Conv_2_Relu_ts[7,3] + Aggregation[3] = 0
 Aggregation_constraint[4]: - 0.125 Conv_2_Relu_ts[0,4]
   - 0.125 Conv_2_Relu_ts[1,4] - 0.125 Conv_2_Relu_ts[2,4]
   - 0.125 Conv_2_Relu_ts[3,4] - 0.125 Conv_2_Relu_ts[4,4]
   - 0.125 Conv_2_Relu_ts[5,4] - 0.125 Conv_2_Relu_ts[6,4]
   - 0.125 Conv_2_Relu_ts[7,4] + Aggregation[4] = 0
 Aggregation_constraint[5]: - 0.125 Conv_2_Relu_ts[0,5]
   - 0.125 Conv_2_Relu_ts[1,5] - 0.125 Conv_2_Relu_ts[2,5]
   - 0.125 Conv_2_Relu_ts[3,5] - 0.125 Conv_2_Relu_ts[4,5]
   - 0.125 Conv_2_Relu_ts[5,5] - 0.125 Conv_2_Relu_ts[6,5]
   - 0.125 Conv_2_Relu_ts[7,5] + Aggregation[5] = 0
 Aggregation_constraint[6]: - 0.125 Conv_2_Relu_ts[0,6]
   - 0.125 Conv_2_Relu_ts[1,6] - 0.125 Conv_2_Relu_ts[2,6]
   - 0.125 Conv_2_Relu_ts[3,6] - 0.125 Conv_2_Relu_ts[4,6]
   - 0.125 Conv_2_Relu_ts[5,6] - 0.125 Conv_2_Relu_ts[6,6]
   - 0.125 Conv_2_Relu_ts[7,6] + Aggregation[6] = 0
 Aggregation_constraint[7]: - 0.125 Conv_2_Relu_ts[0,7]
   - 0.125 Conv_2_Relu_ts[1,7] - 0.125 Conv_2_Relu_ts[2,7]
   - 0.125 Conv_2_Relu_ts[3,7] - 0.125 Conv_2_Relu_ts[4,7]
   - 0.125 Conv_2_Relu_ts[5,7] - 0.125 Conv_2_Relu_ts[6,7]
   - 0.125 Conv_2_Relu_ts[7,7] + Aggregation[7] = 0
 Aggregation_constraint[8]: - 0.125 Conv_2_Relu_ts[0,8]
   - 0.125 Conv_2_Relu_ts[1,8] - 0.125 Conv_2_Relu_ts[2,8]
   - 0.125 Conv_2_Relu_ts[3,8] - 0.125 Conv_2_Relu_ts[4,8]
   - 0.125 Conv_2_Relu_ts[5,8] - 0.125 Conv_2_Relu_ts[6,8]
   - 0.125 Conv_2_Relu_ts[7,8] + Aggregation[8] = 0
 Aggregation_constraint[9]: - 0.125 Conv_2_Relu_ts[0,9]
   - 0.125 Conv_2_Relu_ts[1,9] - 0.125 Conv_2_Relu_ts[2,9]
   - 0.125 Conv_2_Relu_ts[3,9] - 0.125 Conv_2_Relu_ts[4,9]
   - 0.125 Conv_2_Relu_ts[5,9] - 0.125 Conv_2_Relu_ts[6,9]
   - 0.125 Conv_2_Relu_ts[7,9] + Aggregation[9] = 0
 Aggregation_constraint[10]: - 0.125 Conv_2_Relu_ts[0,10]
   - 0.125 Conv_2_Relu_ts[1,10] - 0.125 Conv_2_Relu_ts[2,10]
   - 0.125 Conv_2_Relu_ts[3,10] - 0.125 Conv_2_Relu_ts[4,10]
   - 0.125 Conv_2_Relu_ts[5,10] - 0.125 Conv_2_Relu_ts[6,10]
   - 0.125 Conv_2_Relu_ts[7,10] + Aggregation[10] = 0
 Aggregation_constraint[11]: - 0.125 Conv_2_Relu_ts[0,11]
   - 0.125 Conv_2_Relu_ts[1,11] - 0.125 Conv_2_Relu_ts[2,11]
   - 0.125 Conv_2_Relu_ts[3,11] - 0.125 Conv_2_Relu_ts[4,11]
   - 0.125 Conv_2_Relu_ts[5,11] - 0.125 Conv_2_Relu_ts[6,11]
   - 0.125 Conv_2_Relu_ts[7,11] + Aggregation[11] = 0
 Aggregation_constraint[12]: - 0.125 Conv_2_Relu_ts[0,12]
   - 0.125 Conv_2_Relu_ts[1,12] - 0.125 Conv_2_Relu_ts[2,12]
   - 0.125 Conv_2_Relu_ts[3,12] - 0.125 Conv_2_Relu_ts[4,12]
   - 0.125 Conv_2_Relu_ts[5,12] - 0.125 Conv_2_Relu_ts[6,12]
   - 0.125 Conv_2_Relu_ts[7,12] + Aggregation[12] = 0
 Aggregation_constraint[13]: - 0.125 Conv_2_Relu_ts[0,13]
   - 0.125 Conv_2_Relu_ts[1,13] - 0.125 Conv_2_Relu_ts[2,13]
   - 0.125 Conv_2_Relu_ts[3,13] - 0.125 Conv_2_Relu_ts[4,13]
   - 0.125 Conv_2_Relu_ts[5,13] - 0.125 Conv_2_Relu_ts[6,13]
   - 0.125 Conv_2_Relu_ts[7,13] + Aggregation[13] = 0
 Aggregation_constraint[14]: - 0.125 Conv_2_Relu_ts[0,14]
   - 0.125 Conv_2_Relu_ts[1,14] - 0.125 Conv_2_Relu_ts[2,14]
   - 0.125 Conv_2_Relu_ts[3,14] - 0.125 Conv_2_Relu_ts[4,14]
   - 0.125 Conv_2_Relu_ts[5,14] - 0.125 Conv_2_Relu_ts[6,14]
   - 0.125 Conv_2_Relu_ts[7,14] + Aggregation[14] = 0
 Aggregation_constraint[15]: - 0.125 Conv_2_Relu_ts[0,15]
   - 0.125 Conv_2_Relu_ts[1,15] - 0.125 Conv_2_Relu_ts[2,15]
   - 0.125 Conv_2_Relu_ts[3,15] - 0.125 Conv_2_Relu_ts[4,15]
   - 0.125 Conv_2_Relu_ts[5,15] - 0.125 Conv_2_Relu_ts[6,15]
   - 0.125 Conv_2_Relu_ts[7,15] + Aggregation[15] = 0
 Lin_0_output_constraint[0,0]: 1.48092e-12 Aggregation[1]
   - 1.3236768427304924e-04 Aggregation[3]
   + 3.4432121901772916e-04 Aggregation[6]
   + 0.00439760880544782 Aggregation[7] + 1.59342e-09 Aggregation[8]
   - 4.2956e-08 Aggregation[9] - 1.39238e-11 Aggregation[10]
   - 1.00445e-10 Aggregation[11] - 0.00329855107702315 Aggregation[12]
   - 2.2688624449074268e-04 Aggregation[13]
   + 0.00321260304190218 Aggregation[14] - 1.49215e-07 Aggregation[15]
   + Lin_0_t[0] = -0.00200044037774205
 Lin_0_output_constraint[0,1]: 7.85223e-05 Aggregation[1]
   - 1.99459e-12 Aggregation[2] + 1.54573e-06 Aggregation[3]
   + 8.69546e-05 Aggregation[4] - 3.8505e-10 Aggregation[5]
   + 0.8729893565177917 Aggregation[6] + 0.3209881484508514 Aggregation[7]
   + 1.05658e-07 Aggregation[8] + 1.50563e-05 Aggregation[9]
   + 3.26993e-10 Aggregation[10] + 1.29141e-08 Aggregation[11]
   - 0.9296961426734924 Aggregation[12]
   - 0.0251798368990421 Aggregation[13]
   - 0.3106213212013245 Aggregation[14] - 5.63916e-06 Aggregation[15]
   + Lin_0_t[1] = 0.0768247023224831
 Lin_0_output_constraint[0,2]: 6.74099e-06 Aggregation[1]
   - 1.62433e-11 Aggregation[2] - 1.29848e-06 Aggregation[3]
   - 8.04053e-08 Aggregation[4] + 8.96449e-10 Aggregation[5]
   + 0.2626502215862274 Aggregation[6] - 0.6964917778968811 Aggregation[7]
   - 2.11136e-07 Aggregation[8] - 2.70472e-06 Aggregation[9]
   + 2.43773e-09 Aggregation[10] - 2.58017e-08 Aggregation[11]
   + 0.1755568981170654 Aggregation[12]
   + 0.0274115763604641 Aggregation[13]
   - 0.3965343832969666 Aggregation[14] + 3.94587e-07 Aggregation[15]
   + Lin_0_t[2] = 0.3188049793243408
 Lin_0_output_constraint[0,3]: 8.23778e-06 Aggregation[1]
   - 1.67384e-11 Aggregation[2] + 7.27994e-09 Aggregation[3]
   - 6.3282e-07 Aggregation[4] + 3.11232e-10 Aggregation[5]
   + 0.7962722182273865 Aggregation[6] - 0.6235721707344055 Aggregation[7]
   - 3.16233e-07 Aggregation[8] + 1.85064e-06 Aggregation[9]
   + 1.1906e-09 Aggregation[10] - 6.37068e-09 Aggregation[11]
   + 0.0443844422698021 Aggregation[12]
   + 0.0493672452867031 Aggregation[13]
   - 0.6394777894020081 Aggregation[14] - 3.31886e-06 Aggregation[15]
   + Lin_0_t[3] = 0.4487363696098328
 Lin_0_output_constraint[0,4]: - 3.53085e-05 Aggregation[1]
   + 1.35006e-11 Aggregation[2] - 1.65145e-06 Aggregation[3]
   - 4.705798055510968e-04 Aggregation[4] + 2.2871e-11 Aggregation[5]
   - 1.701683402061462 Aggregation[6] - 0.0426288433372974 Aggregation[7]
   + 7.08227e-08 Aggregation[8] - 1.56638e-05 Aggregation[9]
   - 5.6631e-08 Aggregation[10] - 2.74214e-09 Aggregation[11]
   + 0.3037901222705841 Aggregation[12]
   + 0.00258171698078513 Aggregation[13]
   - 0.2813050150871277 Aggregation[14] + 2.59793e-05 Aggregation[15]
   + Lin_0_t[4] = -0.0249005779623985
 Lin_0_output_constraint[0,5]: - 3.19359e-06 Aggregation[1]
   - 1.19547e-13 Aggregation[2] + 1.83112e-07 Aggregation[3]
   - 6.76497e-09 Aggregation[4] - 1.94939e-11 Aggregation[5]
   + 0.0233058407902718 Aggregation[6] + 0.0614788271486759 Aggregation[7]
   + 4.2001e-08 Aggregation[8] + 3.41615e-07 Aggregation[9]
   + 1.32269e-09 Aggregation[10] + 2.44961e-09 Aggregation[11]
   - 0.1716405749320984 Aggregation[12]
   - 0.0124587072059512 Aggregation[13]
   + 0.1084617450833321 Aggregation[14] - 9.04684e-08 Aggregation[15]
   + Lin_0_t[5] = -0.0279262978583574
 Lin_0_output_constraint[0,6]: - 3.0049e-10 Aggregation[1]
   + 1.32476e-08 Aggregation[3] + 6.22316e-12 Aggregation[4]
   + 3.42293e-05 Aggregation[6] - 9.18518e-05 Aggregation[7]
   + 5.6854e-12 Aggregation[8] + 1.09739e-07 Aggregation[9]
   + 3.45396e-11 Aggregation[11] - 1.29744e-05 Aggregation[12]
   + 7.54765e-07 Aggregation[13] - 4.94512e-05 Aggregation[14]
   + 2.04433e-09 Aggregation[15] + Lin_0_t[6] = 4.59716e-06
 Lin_0_output_constraint[0,7]: - 1.23035e-05 Aggregation[1]
   - 1.57831e-12 Aggregation[2] - 3.07605e-05 Aggregation[3]
   + 7.83066e-13 Aggregation[4] - 3.74631e-10 Aggregation[5]
   - 0.0169283095747232 Aggregation[6] - 0.2245142161846161 Aggregation[7]
   - 2.77722e-08 Aggregation[8] - 6.46546e-06 Aggregation[9]
   + 2.19351e-13 Aggregation[10] - 2.83904e-08 Aggregation[11]
   + 0.1569113731384277 Aggregation[12]
   + 1.1595139949349687e-04 Aggregation[13]
   + 0.0355387143790722 Aggregation[14] + 9.18131e-06 Aggregation[15]
   + Lin_0_t[7] = -0.111688643693924
 Lin_0_Relu_constraint_1[0,0]: - Lin_0_t[0] + Lin_0_Relu_ts[0,0]
   - Lin_0_Relu_ss[0,0] = 0
 Lin_0_Relu_constraint_1[0,1]: - Lin_0_t[1] + Lin_0_Relu_ts[0,1]
   - Lin_0_Relu_ss[0,1] = 0
 Lin_0_Relu_constraint_1[0,2]: - Lin_0_t[2] + Lin_0_Relu_ts[0,2]
   - Lin_0_Relu_ss[0,2] = 0
 Lin_0_Relu_constraint_1[0,3]: - Lin_0_t[3] + Lin_0_Relu_ts[0,3]
   - Lin_0_Relu_ss[0,3] = 0
 Lin_0_Relu_constraint_1[0,4]: - Lin_0_t[4] + Lin_0_Relu_ts[0,4]
   - Lin_0_Relu_ss[0,4] = 0
 Lin_0_Relu_constraint_1[0,5]: - Lin_0_t[5] + Lin_0_Relu_ts[0,5]
   - Lin_0_Relu_ss[0,5] = 0
 Lin_0_Relu_constraint_1[0,6]: - Lin_0_t[6] + Lin_0_Relu_ts[0,6]
   - Lin_0_Relu_ss[0,6] = 0
 Lin_0_Relu_constraint_1[0,7]: - Lin_0_t[7] + Lin_0_Relu_ts[0,7]
   - Lin_0_Relu_ss[0,7] = 0
 Lin_0_Relu_constraint_2[0,0]: Lin_0_Relu_ts[0,0]
   - 0.0646826258068006 C1912 <= 0
 Lin_0_Relu_constraint_2[0,1]: Lin_0_Relu_ts[0,1] - 23.48739610133581 C1913
   <= 0
 Lin_0_Relu_constraint_2[0,2]: Lin_0_Relu_ts[0,2] - 15.10090508869101 C1914
   <= 0
 Lin_0_Relu_constraint_2[0,3]: Lin_0_Relu_ts[0,3] - 18.36376828199457 C1915
   <= 0
 Lin_0_Relu_constraint_2[0,4]: Lin_0_Relu_ts[0,4] - 28.14687881992175 C1916
   <= 0
 Lin_0_Relu_constraint_2[0,5]: Lin_0_Relu_ts[0,5] - 3.366320737461283 C1917
   <= 0
 Lin_0_Relu_constraint_2[0,6]: Lin_0_Relu_ts[0,6] - 0.002196731845088 C1918
   <= 0
 Lin_0_Relu_constraint_2[0,7]: Lin_0_Relu_ts[0,7] - 2.918435776786758 C1919
   <= 0
 Lin_0_Relu_constraint_3[0,0]: Lin_0_Relu_ss[0,0]
   + 0.1120404087063036 C1912 <= 0.1120404087063036
 Lin_0_Relu_constraint_3[0,1]: Lin_0_Relu_ss[0,1] + 15.56777676550227 C1913
   <= 15.56777676550227
 Lin_0_Relu_constraint_3[0,2]: Lin_0_Relu_ss[0,2] + 6.981702896342312 C1914
   <= 6.981702896342312
 Lin_0_Relu_constraint_3[0,3]: Lin_0_Relu_ss[0,3] + 11.59974143443083 C1915
   <= 11.59974143443083
 Lin_0_Relu_constraint_3[0,4]: Lin_0_Relu_ss[0,4] + 5.941669058398669 C1916
   <= 5.941669058398669
 Lin_0_Relu_constraint_3[0,5]: Lin_0_Relu_ss[0,5] + 2.896736603265128 C1917
   <= 2.896736603265128
 Lin_0_Relu_constraint_3[0,6]: Lin_0_Relu_ss[0,6]
   + 4.6017981019739996e-04 C1918 <= 4.6017981019739996e-04
 Lin_0_Relu_constraint_3[0,7]: Lin_0_Relu_ss[0,7] + 3.673082476382535 C1919
   <= 3.673082476382535
 Lin_1_output_constraint[0,0]: 7.964375545270741e-04 Lin_0_Relu_ts[0,0]
   + 0.5862525701522827 Lin_0_Relu_ts[0,1]
   - 0.4726662933826447 Lin_0_Relu_ts[0,2]
   - 0.2809094786643982 Lin_0_Relu_ts[0,3]
   - 0.7185260057449341 Lin_0_Relu_ts[0,4]
   + 0.0163399260491133 Lin_0_Relu_ts[0,5] - 6.30978e-05 Lin_0_Relu_ts[0,6]
   - 0.0198513846844435 Lin_0_Relu_ts[0,7] + Lin_1_t[0]
   = 0.3420023918151855
 Lin_1_output_constraint[0,1]: 1.089e-08 Lin_0_Relu_ts[0,0]
   + 4.37605e-07 Lin_0_Relu_ts[0,1] + 5.43421e-07 Lin_0_Relu_ts[0,2]
   + 4.84852e-07 Lin_0_Relu_ts[0,3] + 7.29533e-06 Lin_0_Relu_ts[0,4]
   + 3.98435e-08 Lin_0_Relu_ts[0,5] - 2.19416e-09 Lin_0_Relu_ts[0,6]
   - 2.21395e-05 Lin_0_Relu_ts[0,7] + Lin_1_t[1] = -8.00669e-06
 Lin_1_output_constraint[0,2]: - 3.13537e-05 Lin_0_Relu_ts[0,0]
   + 0.2076452374458313 Lin_0_Relu_ts[0,1]
   + 0.3428198099136353 Lin_0_Relu_ts[0,2]
   + 0.6404320597648621 Lin_0_Relu_ts[0,3]
   - 1.089619755744934 Lin_0_Relu_ts[0,4]
   + 0.00500678550451994 Lin_0_Relu_ts[0,5]
   + 3.09043e-05 Lin_0_Relu_ts[0,6] - 0.0862867534160614 Lin_0_Relu_ts[0,7]
   + Lin_1_t[2] = -0.0163337141275406
 Lin_1_output_constraint[0,3]: 0.00157637614756823 Lin_0_Relu_ts[0,0]
   - 0.2973600327968597 Lin_0_Relu_ts[0,1]
   + 0.214868426322937 Lin_0_Relu_ts[0,2]
   + 0.2465823888778687 Lin_0_Relu_ts[0,3]
   + 0.0421998649835587 Lin_0_Relu_ts[0,4]
   - 0.0740625336766243 Lin_0_Relu_ts[0,5] + 2.98441e-05 Lin_0_Relu_ts[0,6]
   + 4.642803396563977e-04 Lin_0_Relu_ts[0,7] + Lin_1_t[3]
   = -0.00176100234966725
 Lin_1_output_constraint[0,4]: 0.00311684026382864 Lin_0_Relu_ts[0,0]
   + 0.19813571870327 Lin_0_Relu_ts[0,1]
   - 0.6150757074356079 Lin_0_Relu_ts[0,2]
   - 0.7579656839370728 Lin_0_Relu_ts[0,3]
   + 0.2917612195014954 Lin_0_Relu_ts[0,4]
   + 0.0703688934445381 Lin_0_Relu_ts[0,5] - 7.31415e-05 Lin_0_Relu_ts[0,6]
   - 0.00403159111738205 Lin_0_Relu_ts[0,7] + Lin_1_t[4]
   = 0.262714684009552
 Lin_1_output_constraint[0,5]: - 0.00575445126742125 Lin_0_Relu_ts[0,0]
   - 0.499297708272934 Lin_0_Relu_ts[0,1]
   + 0.4274059534072876 Lin_0_Relu_ts[0,2]
   + 0.3690556883811951 Lin_0_Relu_ts[0,3]
   + 0.0475415624678135 Lin_0_Relu_ts[0,4]
   - 0.1683415621519089 Lin_0_Relu_ts[0,5] + 4.16703e-05 Lin_0_Relu_ts[0,6]
   - 0.010416254401207 Lin_0_Relu_ts[0,7] + Lin_1_t[5]
   = -0.0361219421029091
 Lin_1_output_constraint[0,6]: - 3.69106e-12 Lin_0_Relu_ts[0,0]
   + 2.35226e-10 Lin_0_Relu_ts[0,1] + 3.60219e-09 Lin_0_Relu_ts[0,2]
   + 2.16853e-09 Lin_0_Relu_ts[0,3] + 5.67938e-09 Lin_0_Relu_ts[0,4]
   + 3.49844e-11 Lin_0_Relu_ts[0,5] - 7.06681e-12 Lin_0_Relu_ts[0,6]
   - 2.83279e-06 Lin_0_Relu_ts[0,7] + Lin_1_t[6] = -6.20979e-09
 Lin_1_output_constraint[0,7]: 8.2932924851775169e-04 Lin_0_Relu_ts[0,0]
   - 1.01656174659729 Lin_0_Relu_ts[0,1]
   - 0.1474056243896484 Lin_0_Relu_ts[0,2]
   - 0.700550377368927 Lin_0_Relu_ts[0,3]
   + 1.053641676902771 Lin_0_Relu_ts[0,4]
   - 0.0021451951470226 Lin_0_Relu_ts[0,5] + 4.58083e-05 Lin_0_Relu_ts[0,6]
   + 0.0678508430719376 Lin_0_Relu_ts[0,7] + Lin_1_t[7]
   = 0.138702318072319
 Lin1Relu_constraint_1[0,0]: - Lin_1_t[0] + Lin1Relu_ts[0,0]
   - Lin1Relu_ss[0,0] = 0
 Lin1Relu_constraint_1[0,1]: - Lin_1_t[1] + Lin1Relu_ts[0,1]
   - Lin1Relu_ss[0,1] = 0
 Lin1Relu_constraint_1[0,2]: - Lin_1_t[2] + Lin1Relu_ts[0,2]
   - Lin1Relu_ss[0,2] = 0
 Lin1Relu_constraint_1[0,3]: - Lin_1_t[3] + Lin1Relu_ts[0,3]
   - Lin1Relu_ss[0,3] = 0
 Lin1Relu_constraint_1[0,4]: - Lin_1_t[4] + Lin1Relu_ts[0,4]
   - Lin1Relu_ss[0,4] = 0
 Lin1Relu_constraint_1[0,5]: - Lin_1_t[5] + Lin1Relu_ts[0,5]
   - Lin1Relu_ss[0,5] = 0
 Lin1Relu_constraint_1[0,6]: - Lin_1_t[6] + Lin1Relu_ts[0,6]
   - Lin1Relu_ss[0,6] = 0
 Lin1Relu_constraint_1[0,7]: - Lin_1_t[7] + Lin1Relu_ts[0,7]
   - Lin1Relu_ss[0,7] = 0
 Lin1Relu_constraint_2[0,0]: Lin1Relu_ts[0,0] - 32.57844495195941 C1944
   <= 0
 Lin1Relu_constraint_2[0,1]: Lin1Relu_ts[0,1] - 6.46128e-05 C1945 <= 0
 Lin1Relu_constraint_2[0,2]: Lin1Relu_ts[0,2] - 30.92121960101973 C1946
   <= 0
 Lin1Relu_constraint_2[0,3]: Lin1Relu_ts[0,3] - 7.233531117990599 C1947
   <= 0
 Lin1Relu_constraint_2[0,4]: Lin1Relu_ts[0,4] - 23.21907216629517 C1948
   <= 0
 Lin1Relu_constraint_2[0,5]: Lin1Relu_ts[0,5] - 12.32466612086686 C1949
   <= 0
 Lin1Relu_constraint_2[0,6]: Lin1Relu_ts[0,6] - 8.26732e-06 C1950 <= 0
 Lin1Relu_constraint_2[0,7]: Lin1Relu_ts[0,7] - 38.97431296201979 C1951
   <= 0
 Lin1Relu_constraint_3[0,0]: Lin1Relu_ss[0,0] + 13.82460327817286 C1944
   <= 13.82460327817286
 Lin1Relu_constraint_3[0,1]: Lin1Relu_ss[0,1]
   + 2.3286370404764788e-04 C1945 <= 2.3286370404764788e-04
 Lin1Relu_constraint_3[0,2]: Lin1Relu_ss[0,2] + 21.83153581211795 C1946
   <= 21.83153581211795
 Lin1Relu_constraint_3[0,3]: Lin1Relu_ss[0,3] + 8.962141052203886 C1947
   <= 8.962141052203886
 Lin1Relu_constraint_3[0,4]: Lin1Relu_ss[0,4] + 13.10294566735477 C1948
   <= 13.10294566735477
 Lin1Relu_constraint_3[0,5]: Lin1Relu_ss[0,5] + 14.56961657055903 C1949
   <= 14.56961657055903
 Lin1Relu_constraint_3[0,6]: Lin1Relu_ss[0,6] + 2.59718e-07 C1950
   <= 2.59718e-07
 Lin1Relu_constraint_3[0,7]: Lin1Relu_ss[0,7] + 29.85479667112976 C1951
   <= 29.85479667112976
 Output_output_constraint[0,0]: - 0.7448395490646362 Lin1Relu_ts[0,0]
   + 1.16081e-05 Lin1Relu_ts[0,1] + 0.6486255526542664 Lin1Relu_ts[0,2]
   + 0.00152239156886935 Lin1Relu_ts[0,3]
   - 0.4408386647701263 Lin1Relu_ts[0,4]
   - 0.00698980875313282 Lin1Relu_ts[0,5] - 4.87385e-09 Lin1Relu_ts[0,6]
   + 1.217160582542419 Lin1Relu_ts[0,7] + Output_t[0] = 0.2459999322891235
 Output_output_constraint[0,1]: 0.6199610829353333 Lin1Relu_ts[0,0]
   + 7.50561e-08 Lin1Relu_ts[0,1] - 0.5161861777305603 Lin1Relu_ts[0,2]
   + 0.265589714050293 Lin1Relu_ts[0,3]
   - 0.1124968826770782 Lin1Relu_ts[0,4]
   + 0.5139396190643311 Lin1Relu_ts[0,5] + 1.14897e-09 Lin1Relu_ts[0,6]
   - 0.7659099698066711 Lin1Relu_ts[0,7] + Output_t[1]
   = -0.0746068805456161
 Output_output_constraint[0,2]: - 0.3719607293605804 Lin1Relu_ts[0,0]
   + 1.31159e-06 Lin1Relu_ts[0,1] - 0.8342773914337158 Lin1Relu_ts[0,2]
   + 0.00765973050147295 Lin1Relu_ts[0,3]
   + 0.5495966076850891 Lin1Relu_ts[0,4]
   - 0.0118089355528355 Lin1Relu_ts[0,5] + 4.31919e-09 Lin1Relu_ts[0,6]
   + 0.3341525793075562 Lin1Relu_ts[0,7] + Output_t[2]
   = 0.0628847479820251
 Output_output_constraint[0,3]: 0.5905329585075378 Lin1Relu_ts[0,0]
   + 3.04252e-08 Lin1Relu_ts[0,1] + 0.3330641686916351 Lin1Relu_ts[0,2]
   - 0.3092302680015564 Lin1Relu_ts[0,3]
   + 0.6715567708015442 Lin1Relu_ts[0,4]
   - 0.5544244647026062 Lin1Relu_ts[0,5] + 4.02779e-11 Lin1Relu_ts[0,6]
   - 0.5177575945854187 Lin1Relu_ts[0,7] + Output_t[3]
   = -0.2988127470016479
 Output_output_constraint[0,4]: 0.0145479794591665 Lin1Relu_ts[0,0]
   - 1.21104e-05 Lin1Relu_ts[0,1] + 0.3993335664272308 Lin1Relu_ts[0,2]
   + 0.0158467423170805 Lin1Relu_ts[0,3]
   - 0.6261000037193298 Lin1Relu_ts[0,4]
   + 0.0431621074676514 Lin1Relu_ts[0,5] - 6.60263e-09 Lin1Relu_ts[0,6]
   - 0.2009809166193008 Lin1Relu_ts[0,7] + Output_t[4] = 0.006735616363585
 Conv_0_averages_constraint[0,0]: [ - A[0,0] * X[0,0]
   + A[0,0] * Conv_0_aggregated_features[0,0] - A[0,1] * X[1,0]
   - A[0,2] * X[2,0] - A[0,3] * X[3,0] - A[0,4] * X[4,0] - A[0,5] * X[5,0]
   - A[0,6] * X[6,0] - A[0,7] * X[7,0]
   + A[1,0] * Conv_0_aggregated_features[0,0]
   + A[2,0] * Conv_0_aggregated_features[0,0]
   + A[3,0] * Conv_0_aggregated_features[0,0]
   + A[4,0] * Conv_0_aggregated_features[0,0]
   + A[5,0] * Conv_0_aggregated_features[0,0]
   + A[6,0] * Conv_0_aggregated_features[0,0]
   + A[7,0] * Conv_0_aggregated_features[0,0] ] = 0
 Conv_0_averages_constraint[1,0]: [
   A[0,1] * Conv_0_aggregated_features[1,0] - A[1,0] * X[0,0]
   - A[1,1] * X[1,0] + A[1,1] * Conv_0_aggregated_features[1,0]
   - A[1,2] * X[2,0] - A[1,3] * X[3,0] - A[1,4] * X[4,0] - A[1,5] * X[5,0]
   - A[1,6] * X[6,0] - A[1,7] * X[7,0]
   + A[2,1] * Conv_0_aggregated_features[1,0]
   + A[3,1] * Conv_0_aggregated_features[1,0]
   + A[4,1] * Conv_0_aggregated_features[1,0]
   + A[5,1] * Conv_0_aggregated_features[1,0]
   + A[6,1] * Conv_0_aggregated_features[1,0]
   + A[7,1] * Conv_0_aggregated_features[1,0] ] = 0
 Conv_0_averages_constraint[2,0]: [
   A[0,2] * Conv_0_aggregated_features[2,0]
   + A[1,2] * Conv_0_aggregated_features[2,0] - A[2,0] * X[0,0]
   - A[2,1] * X[1,0] - A[2,2] * X[2,0]
   + A[2,2] * Conv_0_aggregated_features[2,0] - A[2,3] * X[3,0]
   - A[2,4] * X[4,0] - A[2,5] * X[5,0] - A[2,6] * X[6,0] - A[2,7] * X[7,0]
   + A[3,2] * Conv_0_aggregated_features[2,0]
   + A[4,2] * Conv_0_aggregated_features[2,0]
   + A[5,2] * Conv_0_aggregated_features[2,0]
   + A[6,2] * Conv_0_aggregated_features[2,0]
   + A[7,2] * Conv_0_aggregated_features[2,0] ] = 0
 Conv_0_averages_constraint[3,0]: [
   A[0,3] * Conv_0_aggregated_features[3,0]
   + A[1,3] * Conv_0_aggregated_features[3,0]
   + A[2,3] * Conv_0_aggregated_features[3,0] - A[3,0] * X[0,0]
   - A[3,1] * X[1,0] - A[3,2] * X[2,0] - A[3,3] * X[3,0]
   + A[3,3] * Conv_0_aggregated_features[3,0] - A[3,4] * X[4,0]
   - A[3,5] * X[5,0] - A[3,6] * X[6,0] - A[3,7] * X[7,0]
   + A[4,3] * Conv_0_aggregated_features[3,0]
   + A[5,3] * Conv_0_aggregated_features[3,0]
   + A[6,3] * Conv_0_aggregated_features[3,0]
   + A[7,3] * Conv_0_aggregated_features[3,0] ] = 0
 Conv_0_averages_constraint[4,0]: [
   A[0,4] * Conv_0_aggregated_features[4,0]
   + A[1,4] * Conv_0_aggregated_features[4,0]
   + A[2,4] * Conv_0_aggregated_features[4,0]
   + A[3,4] * Conv_0_aggregated_features[4,0] - A[4,0] * X[0,0]
   - A[4,1] * X[1,0] - A[4,2] * X[2,0] - A[4,3] * X[3,0] - A[4,4] * X[4,0]
   + A[4,4] * Conv_0_aggregated_features[4,0] - A[4,5] * X[5,0]
   - A[4,6] * X[6,0] - A[4,7] * X[7,0]
   + A[5,4] * Conv_0_aggregated_features[4,0]
   + A[6,4] * Conv_0_aggregated_features[4,0]
   + A[7,4] * Conv_0_aggregated_features[4,0] ] = 0
 Conv_0_averages_constraint[5,0]: [
   A[0,5] * Conv_0_aggregated_features[5,0]
   + A[1,5] * Conv_0_aggregated_features[5,0]
   + A[2,5] * Conv_0_aggregated_features[5,0]
   + A[3,5] * Conv_0_aggregated_features[5,0]
   + A[4,5] * Conv_0_aggregated_features[5,0] - A[5,0] * X[0,0]
   - A[5,1] * X[1,0] - A[5,2] * X[2,0] - A[5,3] * X[3,0] - A[5,4] * X[4,0]
   - A[5,5] * X[5,0] + A[5,5] * Conv_0_aggregated_features[5,0]
   - A[5,6] * X[6,0] - A[5,7] * X[7,0]
   + A[6,5] * Conv_0_aggregated_features[5,0]
   + A[7,5] * Conv_0_aggregated_features[5,0] ] = 0
 Conv_0_averages_constraint[6,0]: [
   A[0,6] * Conv_0_aggregated_features[6,0]
   + A[1,6] * Conv_0_aggregated_features[6,0]
   + A[2,6] * Conv_0_aggregated_features[6,0]
   + A[3,6] * Conv_0_aggregated_features[6,0]
   + A[4,6] * Conv_0_aggregated_features[6,0]
   + A[5,6] * Conv_0_aggregated_features[6,0] - A[6,0] * X[0,0]
   - A[6,1] * X[1,0] - A[6,2] * X[2,0] - A[6,3] * X[3,0] - A[6,4] * X[4,0]
   - A[6,5] * X[5,0] - A[6,6] * X[6,0]
   + A[6,6] * Conv_0_aggregated_features[6,0] - A[6,7] * X[7,0]
   + A[7,6] * Conv_0_aggregated_features[6,0] ] = 0
 Conv_0_averages_constraint[7,0]: [
   A[0,7] * Conv_0_aggregated_features[7,0]
   + A[1,7] * Conv_0_aggregated_features[7,0]
   + A[2,7] * Conv_0_aggregated_features[7,0]
   + A[3,7] * Conv_0_aggregated_features[7,0]
   + A[4,7] * Conv_0_aggregated_features[7,0]
   + A[5,7] * Conv_0_aggregated_features[7,0]
   + A[6,7] * Conv_0_aggregated_features[7,0] - A[7,0] * X[0,0]
   - A[7,1] * X[1,0] - A[7,2] * X[2,0] - A[7,3] * X[3,0] - A[7,4] * X[4,0]
   - A[7,5] * X[5,0] - A[7,6] * X[6,0] - A[7,7] * X[7,0]
   + A[7,7] * Conv_0_aggregated_features[7,0] ] = 0
 Conv_1_averages_constraint[0,0]: [ - A[0,0] * Conv_0_Relu_ts[0,0]
   + A[0,0] * Conv_1_aggregated_features[0,0]
   - A[0,1] * Conv_0_Relu_ts[1,0] - A[0,2] * Conv_0_Relu_ts[2,0]
   - A[0,3] * Conv_0_Relu_ts[3,0] - A[0,4] * Conv_0_Relu_ts[4,0]
   - A[0,5] * Conv_0_Relu_ts[5,0] - A[0,6] * Conv_0_Relu_ts[6,0]
   - A[0,7] * Conv_0_Relu_ts[7,0]
   + A[1,0] * Conv_1_aggregated_features[0,0]
   + A[2,0] * Conv_1_aggregated_features[0,0]
   + A[3,0] * Conv_1_aggregated_features[0,0]
   + A[4,0] * Conv_1_aggregated_features[0,0]
   + A[5,0] * Conv_1_aggregated_features[0,0]
   + A[6,0] * Conv_1_aggregated_features[0,0]
   + A[7,0] * Conv_1_aggregated_features[0,0] ] = 0
 Conv_1_averages_constraint[0,1]: [ - A[0,0] * Conv_0_Relu_ts[0,1]
   + A[0,0] * Conv_1_aggregated_features[0,1]
   - A[0,1] * Conv_0_Relu_ts[1,1] - A[0,2] * Conv_0_Relu_ts[2,1]
   - A[0,3] * Conv_0_Relu_ts[3,1] - A[0,4] * Conv_0_Relu_ts[4,1]
   - A[0,5] * Conv_0_Relu_ts[5,1] - A[0,6] * Conv_0_Relu_ts[6,1]
   - A[0,7] * Conv_0_Relu_ts[7,1]
   + A[1,0] * Conv_1_aggregated_features[0,1]
   + A[2,0] * Conv_1_aggregated_features[0,1]
   + A[3,0] * Conv_1_aggregated_features[0,1]
   + A[4,0] * Conv_1_aggregated_features[0,1]
   + A[5,0] * Conv_1_aggregated_features[0,1]
   + A[6,0] * Conv_1_aggregated_features[0,1]
   + A[7,0] * Conv_1_aggregated_features[0,1] ] = 0
 Conv_1_averages_constraint[0,2]: [ - A[0,0] * Conv_0_Relu_ts[0,2]
   + A[0,0] * Conv_1_aggregated_features[0,2]
   - A[0,1] * Conv_0_Relu_ts[1,2] - A[0,2] * Conv_0_Relu_ts[2,2]
   - A[0,3] * Conv_0_Relu_ts[3,2] - A[0,4] * Conv_0_Relu_ts[4,2]
   - A[0,5] * Conv_0_Relu_ts[5,2] - A[0,6] * Conv_0_Relu_ts[6,2]
   - A[0,7] * Conv_0_Relu_ts[7,2]
   + A[1,0] * Conv_1_aggregated_features[0,2]
   + A[2,0] * Conv_1_aggregated_features[0,2]
   + A[3,0] * Conv_1_aggregated_features[0,2]
   + A[4,0] * Conv_1_aggregated_features[0,2]
   + A[5,0] * Conv_1_aggregated_features[0,2]
   + A[6,0] * Conv_1_aggregated_features[0,2]
   + A[7,0] * Conv_1_aggregated_features[0,2] ] = 0
 Conv_1_averages_constraint[0,3]: [ - A[0,0] * Conv_0_Relu_ts[0,3]
   + A[0,0] * Conv_1_aggregated_features[0,3]
   - A[0,1] * Conv_0_Relu_ts[1,3] - A[0,2] * Conv_0_Relu_ts[2,3]
   - A[0,3] * Conv_0_Relu_ts[3,3] - A[0,4] * Conv_0_Relu_ts[4,3]
   - A[0,5] * Conv_0_Relu_ts[5,3] - A[0,6] * Conv_0_Relu_ts[6,3]
   - A[0,7] * Conv_0_Relu_ts[7,3]
   + A[1,0] * Conv_1_aggregated_features[0,3]
   + A[2,0] * Conv_1_aggregated_features[0,3]
   + A[3,0] * Conv_1_aggregated_features[0,3]
   + A[4,0] * Conv_1_aggregated_features[0,3]
   + A[5,0] * Conv_1_aggregated_features[0,3]
   + A[6,0] * Conv_1_aggregated_features[0,3]
   + A[7,0] * Conv_1_aggregated_features[0,3] ] = 0
 Conv_1_averages_constraint[0,4]: [ - A[0,0] * Conv_0_Relu_ts[0,4]
   + A[0,0] * Conv_1_aggregated_features[0,4]
   - A[0,1] * Conv_0_Relu_ts[1,4] - A[0,2] * Conv_0_Relu_ts[2,4]
   - A[0,3] * Conv_0_Relu_ts[3,4] - A[0,4] * Conv_0_Relu_ts[4,4]
   - A[0,5] * Conv_0_Relu_ts[5,4] - A[0,6] * Conv_0_Relu_ts[6,4]
   - A[0,7] * Conv_0_Relu_ts[7,4]
   + A[1,0] * Conv_1_aggregated_features[0,4]
   + A[2,0] * Conv_1_aggregated_features[0,4]
   + A[3,0] * Conv_1_aggregated_features[0,4]
   + A[4,0] * Conv_1_aggregated_features[0,4]
   + A[5,0] * Conv_1_aggregated_features[0,4]
   + A[6,0] * Conv_1_aggregated_features[0,4]
   + A[7,0] * Conv_1_aggregated_features[0,4] ] = 0
 Conv_1_averages_constraint[0,5]: [ - A[0,0] * Conv_0_Relu_ts[0,5]
   + A[0,0] * Conv_1_aggregated_features[0,5]
   - A[0,1] * Conv_0_Relu_ts[1,5] - A[0,2] * Conv_0_Relu_ts[2,5]
   - A[0,3] * Conv_0_Relu_ts[3,5] - A[0,4] * Conv_0_Relu_ts[4,5]
   - A[0,5] * Conv_0_Relu_ts[5,5] - A[0,6] * Conv_0_Relu_ts[6,5]
   - A[0,7] * Conv_0_Relu_ts[7,5]
   + A[1,0] * Conv_1_aggregated_features[0,5]
   + A[2,0] * Conv_1_aggregated_features[0,5]
   + A[3,0] * Conv_1_aggregated_features[0,5]
   + A[4,0] * Conv_1_aggregated_features[0,5]
   + A[5,0] * Conv_1_aggregated_features[0,5]
   + A[6,0] * Conv_1_aggregated_features[0,5]
   + A[7,0] * Conv_1_aggregated_features[0,5] ] = 0
 Conv_1_averages_constraint[0,6]: [ - A[0,0] * Conv_0_Relu_ts[0,6]
   + A[0,0] * Conv_1_aggregated_features[0,6]
   - A[0,1] * Conv_0_Relu_ts[1,6] - A[0,2] * Conv_0_Relu_ts[2,6]
   - A[0,3] * Conv_0_Relu_ts[3,6] - A[0,4] * Conv_0_Relu_ts[4,6]
   - A[0,5] * Conv_0_Relu_ts[5,6] - A[0,6] * Conv_0_Relu_ts[6,6]
   - A[0,7] * Conv_0_Relu_ts[7,6]
   + A[1,0] * Conv_1_aggregated_features[0,6]
   + A[2,0] * Conv_1_aggregated_features[0,6]
   + A[3,0] * Conv_1_aggregated_features[0,6]
   + A[4,0] * Conv_1_aggregated_features[0,6]
   + A[5,0] * Conv_1_aggregated_features[0,6]
   + A[6,0] * Conv_1_aggregated_features[0,6]
   + A[7,0] * Conv_1_aggregated_features[0,6] ] = 0
 Conv_1_averages_constraint[0,7]: [ - A[0,0] * Conv_0_Relu_ts[0,7]
   + A[0,0] * Conv_1_aggregated_features[0,7]
   - A[0,1] * Conv_0_Relu_ts[1,7] - A[0,2] * Conv_0_Relu_ts[2,7]
   - A[0,3] * Conv_0_Relu_ts[3,7] - A[0,4] * Conv_0_Relu_ts[4,7]
   - A[0,5] * Conv_0_Relu_ts[5,7] - A[0,6] * Conv_0_Relu_ts[6,7]
   - A[0,7] * Conv_0_Relu_ts[7,7]
   + A[1,0] * Conv_1_aggregated_features[0,7]
   + A[2,0] * Conv_1_aggregated_features[0,7]
   + A[3,0] * Conv_1_aggregated_features[0,7]
   + A[4,0] * Conv_1_aggregated_features[0,7]
   + A[5,0] * Conv_1_aggregated_features[0,7]
   + A[6,0] * Conv_1_aggregated_features[0,7]
   + A[7,0] * Conv_1_aggregated_features[0,7] ] = 0
 Conv_1_averages_constraint[0,8]: [ - A[0,0] * Conv_0_Relu_ts[0,8]
   + A[0,0] * Conv_1_aggregated_features[0,8]
   - A[0,1] * Conv_0_Relu_ts[1,8] - A[0,2] * Conv_0_Relu_ts[2,8]
   - A[0,3] * Conv_0_Relu_ts[3,8] - A[0,4] * Conv_0_Relu_ts[4,8]
   - A[0,5] * Conv_0_Relu_ts[5,8] - A[0,6] * Conv_0_Relu_ts[6,8]
   - A[0,7] * Conv_0_Relu_ts[7,8]
   + A[1,0] * Conv_1_aggregated_features[0,8]
   + A[2,0] * Conv_1_aggregated_features[0,8]
   + A[3,0] * Conv_1_aggregated_features[0,8]
   + A[4,0] * Conv_1_aggregated_features[0,8]
   + A[5,0] * Conv_1_aggregated_features[0,8]
   + A[6,0] * Conv_1_aggregated_features[0,8]
   + A[7,0] * Conv_1_aggregated_features[0,8] ] = 0
 Conv_1_averages_constraint[0,9]: [ - A[0,0] * Conv_0_Relu_ts[0,9]
   + A[0,0] * Conv_1_aggregated_features[0,9]
   - A[0,1] * Conv_0_Relu_ts[1,9] - A[0,2] * Conv_0_Relu_ts[2,9]
   - A[0,3] * Conv_0_Relu_ts[3,9] - A[0,4] * Conv_0_Relu_ts[4,9]
   - A[0,5] * Conv_0_Relu_ts[5,9] - A[0,6] * Conv_0_Relu_ts[6,9]
   - A[0,7] * Conv_0_Relu_ts[7,9]
   + A[1,0] * Conv_1_aggregated_features[0,9]
   + A[2,0] * Conv_1_aggregated_features[0,9]
   + A[3,0] * Conv_1_aggregated_features[0,9]
   + A[4,0] * Conv_1_aggregated_features[0,9]
   + A[5,0] * Conv_1_aggregated_features[0,9]
   + A[6,0] * Conv_1_aggregated_features[0,9]
   + A[7,0] * Conv_1_aggregated_features[0,9] ] = 0
 Conv_1_averages_constraint[0,10]: [ - A[0,0] * Conv_0_Relu_ts[0,10]
   + A[0,0] * Conv_1_aggregated_features[0,10]
   - A[0,1] * Conv_0_Relu_ts[1,10] - A[0,2] * Conv_0_Relu_ts[2,10]
   - A[0,3] * Conv_0_Relu_ts[3,10] - A[0,4] * Conv_0_Relu_ts[4,10]
   - A[0,5] * Conv_0_Relu_ts[5,10] - A[0,6] * Conv_0_Relu_ts[6,10]
   - A[0,7] * Conv_0_Relu_ts[7,10]
   + A[1,0] * Conv_1_aggregated_features[0,10]
   + A[2,0] * Conv_1_aggregated_features[0,10]
   + A[3,0] * Conv_1_aggregated_features[0,10]
   + A[4,0] * Conv_1_aggregated_features[0,10]
   + A[5,0] * Conv_1_aggregated_features[0,10]
   + A[6,0] * Conv_1_aggregated_features[0,10]
   + A[7,0] * Conv_1_aggregated_features[0,10] ] = 0
 Conv_1_averages_constraint[0,11]: [ - A[0,0] * Conv_0_Relu_ts[0,11]
   + A[0,0] * Conv_1_aggregated_features[0,11]
   - A[0,1] * Conv_0_Relu_ts[1,11] - A[0,2] * Conv_0_Relu_ts[2,11]
   - A[0,3] * Conv_0_Relu_ts[3,11] - A[0,4] * Conv_0_Relu_ts[4,11]
   - A[0,5] * Conv_0_Relu_ts[5,11] - A[0,6] * Conv_0_Relu_ts[6,11]
   - A[0,7] * Conv_0_Relu_ts[7,11]
   + A[1,0] * Conv_1_aggregated_features[0,11]
   + A[2,0] * Conv_1_aggregated_features[0,11]
   + A[3,0] * Conv_1_aggregated_features[0,11]
   + A[4,0] * Conv_1_aggregated_features[0,11]
   + A[5,0] * Conv_1_aggregated_features[0,11]
   + A[6,0] * Conv_1_aggregated_features[0,11]
   + A[7,0] * Conv_1_aggregated_features[0,11] ] = 0
 Conv_1_averages_constraint[0,12]: [ - A[0,0] * Conv_0_Relu_ts[0,12]
   + A[0,0] * Conv_1_aggregated_features[0,12]
   - A[0,1] * Conv_0_Relu_ts[1,12] - A[0,2] * Conv_0_Relu_ts[2,12]
   - A[0,3] * Conv_0_Relu_ts[3,12] - A[0,4] * Conv_0_Relu_ts[4,12]
   - A[0,5] * Conv_0_Relu_ts[5,12] - A[0,6] * Conv_0_Relu_ts[6,12]
   - A[0,7] * Conv_0_Relu_ts[7,12]
   + A[1,0] * Conv_1_aggregated_features[0,12]
   + A[2,0] * Conv_1_aggregated_features[0,12]
   + A[3,0] * Conv_1_aggregated_features[0,12]
   + A[4,0] * Conv_1_aggregated_features[0,12]
   + A[5,0] * Conv_1_aggregated_features[0,12]
   + A[6,0] * Conv_1_aggregated_features[0,12]
   + A[7,0] * Conv_1_aggregated_features[0,12] ] = 0
 Conv_1_averages_constraint[0,13]: [ - A[0,0] * Conv_0_Relu_ts[0,13]
   + A[0,0] * Conv_1_aggregated_features[0,13]
   - A[0,1] * Conv_0_Relu_ts[1,13] - A[0,2] * Conv_0_Relu_ts[2,13]
   - A[0,3] * Conv_0_Relu_ts[3,13] - A[0,4] * Conv_0_Relu_ts[4,13]
   - A[0,5] * Conv_0_Relu_ts[5,13] - A[0,6] * Conv_0_Relu_ts[6,13]
   - A[0,7] * Conv_0_Relu_ts[7,13]
   + A[1,0] * Conv_1_aggregated_features[0,13]
   + A[2,0] * Conv_1_aggregated_features[0,13]
   + A[3,0] * Conv_1_aggregated_features[0,13]
   + A[4,0] * Conv_1_aggregated_features[0,13]
   + A[5,0] * Conv_1_aggregated_features[0,13]
   + A[6,0] * Conv_1_aggregated_features[0,13]
   + A[7,0] * Conv_1_aggregated_features[0,13] ] = 0
 Conv_1_averages_constraint[0,14]: [ - A[0,0] * Conv_0_Relu_ts[0,14]
   + A[0,0] * Conv_1_aggregated_features[0,14]
   - A[0,1] * Conv_0_Relu_ts[1,14] - A[0,2] * Conv_0_Relu_ts[2,14]
   - A[0,3] * Conv_0_Relu_ts[3,14] - A[0,4] * Conv_0_Relu_ts[4,14]
   - A[0,5] * Conv_0_Relu_ts[5,14] - A[0,6] * Conv_0_Relu_ts[6,14]
   - A[0,7] * Conv_0_Relu_ts[7,14]
   + A[1,0] * Conv_1_aggregated_features[0,14]
   + A[2,0] * Conv_1_aggregated_features[0,14]
   + A[3,0] * Conv_1_aggregated_features[0,14]
   + A[4,0] * Conv_1_aggregated_features[0,14]
   + A[5,0] * Conv_1_aggregated_features[0,14]
   + A[6,0] * Conv_1_aggregated_features[0,14]
   + A[7,0] * Conv_1_aggregated_features[0,14] ] = 0
 Conv_1_averages_constraint[0,15]: [ - A[0,0] * Conv_0_Relu_ts[0,15]
   + A[0,0] * Conv_1_aggregated_features[0,15]
   - A[0,1] * Conv_0_Relu_ts[1,15] - A[0,2] * Conv_0_Relu_ts[2,15]
   - A[0,3] * Conv_0_Relu_ts[3,15] - A[0,4] * Conv_0_Relu_ts[4,15]
   - A[0,5] * Conv_0_Relu_ts[5,15] - A[0,6] * Conv_0_Relu_ts[6,15]
   - A[0,7] * Conv_0_Relu_ts[7,15]
   + A[1,0] * Conv_1_aggregated_features[0,15]
   + A[2,0] * Conv_1_aggregated_features[0,15]
   + A[3,0] * Conv_1_aggregated_features[0,15]
   + A[4,0] * Conv_1_aggregated_features[0,15]
   + A[5,0] * Conv_1_aggregated_features[0,15]
   + A[6,0] * Conv_1_aggregated_features[0,15]
   + A[7,0] * Conv_1_aggregated_features[0,15] ] = 0
 Conv_1_averages_constraint[1,0]: [
   A[0,1] * Conv_1_aggregated_features[1,0] - A[1,0] * Conv_0_Relu_ts[0,0]
   - A[1,1] * Conv_0_Relu_ts[1,0]
   + A[1,1] * Conv_1_aggregated_features[1,0]
   - A[1,2] * Conv_0_Relu_ts[2,0] - A[1,3] * Conv_0_Relu_ts[3,0]
   - A[1,4] * Conv_0_Relu_ts[4,0] - A[1,5] * Conv_0_Relu_ts[5,0]
   - A[1,6] * Conv_0_Relu_ts[6,0] - A[1,7] * Conv_0_Relu_ts[7,0]
   + A[2,1] * Conv_1_aggregated_features[1,0]
   + A[3,1] * Conv_1_aggregated_features[1,0]
   + A[4,1] * Conv_1_aggregated_features[1,0]
   + A[5,1] * Conv_1_aggregated_features[1,0]
   + A[6,1] * Conv_1_aggregated_features[1,0]
   + A[7,1] * Conv_1_aggregated_features[1,0] ] = 0
 Conv_1_averages_constraint[1,1]: [
   A[0,1] * Conv_1_aggregated_features[1,1] - A[1,0] * Conv_0_Relu_ts[0,1]
   - A[1,1] * Conv_0_Relu_ts[1,1]
   + A[1,1] * Conv_1_aggregated_features[1,1]
   - A[1,2] * Conv_0_Relu_ts[2,1] - A[1,3] * Conv_0_Relu_ts[3,1]
   - A[1,4] * Conv_0_Relu_ts[4,1] - A[1,5] * Conv_0_Relu_ts[5,1]
   - A[1,6] * Conv_0_Relu_ts[6,1] - A[1,7] * Conv_0_Relu_ts[7,1]
   + A[2,1] * Conv_1_aggregated_features[1,1]
   + A[3,1] * Conv_1_aggregated_features[1,1]
   + A[4,1] * Conv_1_aggregated_features[1,1]
   + A[5,1] * Conv_1_aggregated_features[1,1]
   + A[6,1] * Conv_1_aggregated_features[1,1]
   + A[7,1] * Conv_1_aggregated_features[1,1] ] = 0
 Conv_1_averages_constraint[1,2]: [
   A[0,1] * Conv_1_aggregated_features[1,2] - A[1,0] * Conv_0_Relu_ts[0,2]
   - A[1,1] * Conv_0_Relu_ts[1,2]
   + A[1,1] * Conv_1_aggregated_features[1,2]
   - A[1,2] * Conv_0_Relu_ts[2,2] - A[1,3] * Conv_0_Relu_ts[3,2]
   - A[1,4] * Conv_0_Relu_ts[4,2] - A[1,5] * Conv_0_Relu_ts[5,2]
   - A[1,6] * Conv_0_Relu_ts[6,2] - A[1,7] * Conv_0_Relu_ts[7,2]
   + A[2,1] * Conv_1_aggregated_features[1,2]
   + A[3,1] * Conv_1_aggregated_features[1,2]
   + A[4,1] * Conv_1_aggregated_features[1,2]
   + A[5,1] * Conv_1_aggregated_features[1,2]
   + A[6,1] * Conv_1_aggregated_features[1,2]
   + A[7,1] * Conv_1_aggregated_features[1,2] ] = 0
 Conv_1_averages_constraint[1,3]: [
   A[0,1] * Conv_1_aggregated_features[1,3] - A[1,0] * Conv_0_Relu_ts[0,3]
   - A[1,1] * Conv_0_Relu_ts[1,3]
   + A[1,1] * Conv_1_aggregated_features[1,3]
   - A[1,2] * Conv_0_Relu_ts[2,3] - A[1,3] * Conv_0_Relu_ts[3,3]
   - A[1,4] * Conv_0_Relu_ts[4,3] - A[1,5] * Conv_0_Relu_ts[5,3]
   - A[1,6] * Conv_0_Relu_ts[6,3] - A[1,7] * Conv_0_Relu_ts[7,3]
   + A[2,1] * Conv_1_aggregated_features[1,3]
   + A[3,1] * Conv_1_aggregated_features[1,3]
   + A[4,1] * Conv_1_aggregated_features[1,3]
   + A[5,1] * Conv_1_aggregated_features[1,3]
   + A[6,1] * Conv_1_aggregated_features[1,3]
   + A[7,1] * Conv_1_aggregated_features[1,3] ] = 0
 Conv_1_averages_constraint[1,4]: [
   A[0,1] * Conv_1_aggregated_features[1,4] - A[1,0] * Conv_0_Relu_ts[0,4]
   - A[1,1] * Conv_0_Relu_ts[1,4]
   + A[1,1] * Conv_1_aggregated_features[1,4]
   - A[1,2] * Conv_0_Relu_ts[2,4] - A[1,3] * Conv_0_Relu_ts[3,4]
   - A[1,4] * Conv_0_Relu_ts[4,4] - A[1,5] * Conv_0_Relu_ts[5,4]
   - A[1,6] * Conv_0_Relu_ts[6,4] - A[1,7] * Conv_0_Relu_ts[7,4]
   + A[2,1] * Conv_1_aggregated_features[1,4]
   + A[3,1] * Conv_1_aggregated_features[1,4]
   + A[4,1] * Conv_1_aggregated_features[1,4]
   + A[5,1] * Conv_1_aggregated_features[1,4]
   + A[6,1] * Conv_1_aggregated_features[1,4]
   + A[7,1] * Conv_1_aggregated_features[1,4] ] = 0
 Conv_1_averages_constraint[1,5]: [
   A[0,1] * Conv_1_aggregated_features[1,5] - A[1,0] * Conv_0_Relu_ts[0,5]
   - A[1,1] * Conv_0_Relu_ts[1,5]
   + A[1,1] * Conv_1_aggregated_features[1,5]
   - A[1,2] * Conv_0_Relu_ts[2,5] - A[1,3] * Conv_0_Relu_ts[3,5]
   - A[1,4] * Conv_0_Relu_ts[4,5] - A[1,5] * Conv_0_Relu_ts[5,5]
   - A[1,6] * Conv_0_Relu_ts[6,5] - A[1,7] * Conv_0_Relu_ts[7,5]
   + A[2,1] * Conv_1_aggregated_features[1,5]
   + A[3,1] * Conv_1_aggregated_features[1,5]
   + A[4,1] * Conv_1_aggregated_features[1,5]
   + A[5,1] * Conv_1_aggregated_features[1,5]
   + A[6,1] * Conv_1_aggregated_features[1,5]
   + A[7,1] * Conv_1_aggregated_features[1,5] ] = 0
 Conv_1_averages_constraint[1,6]: [
   A[0,1] * Conv_1_aggregated_features[1,6] - A[1,0] * Conv_0_Relu_ts[0,6]
   - A[1,1] * Conv_0_Relu_ts[1,6]
   + A[1,1] * Conv_1_aggregated_features[1,6]
   - A[1,2] * Conv_0_Relu_ts[2,6] - A[1,3] * Conv_0_Relu_ts[3,6]
   - A[1,4] * Conv_0_Relu_ts[4,6] - A[1,5] * Conv_0_Relu_ts[5,6]
   - A[1,6] * Conv_0_Relu_ts[6,6] - A[1,7] * Conv_0_Relu_ts[7,6]
   + A[2,1] * Conv_1_aggregated_features[1,6]
   + A[3,1] * Conv_1_aggregated_features[1,6]
   + A[4,1] * Conv_1_aggregated_features[1,6]
   + A[5,1] * Conv_1_aggregated_features[1,6]
   + A[6,1] * Conv_1_aggregated_features[1,6]
   + A[7,1] * Conv_1_aggregated_features[1,6] ] = 0
 Conv_1_averages_constraint[1,7]: [
   A[0,1] * Conv_1_aggregated_features[1,7] - A[1,0] * Conv_0_Relu_ts[0,7]
   - A[1,1] * Conv_0_Relu_ts[1,7]
   + A[1,1] * Conv_1_aggregated_features[1,7]
   - A[1,2] * Conv_0_Relu_ts[2,7] - A[1,3] * Conv_0_Relu_ts[3,7]
   - A[1,4] * Conv_0_Relu_ts[4,7] - A[1,5] * Conv_0_Relu_ts[5,7]
   - A[1,6] * Conv_0_Relu_ts[6,7] - A[1,7] * Conv_0_Relu_ts[7,7]
   + A[2,1] * Conv_1_aggregated_features[1,7]
   + A[3,1] * Conv_1_aggregated_features[1,7]
   + A[4,1] * Conv_1_aggregated_features[1,7]
   + A[5,1] * Conv_1_aggregated_features[1,7]
   + A[6,1] * Conv_1_aggregated_features[1,7]
   + A[7,1] * Conv_1_aggregated_features[1,7] ] = 0
 Conv_1_averages_constraint[1,8]: [
   A[0,1] * Conv_1_aggregated_features[1,8] - A[1,0] * Conv_0_Relu_ts[0,8]
   - A[1,1] * Conv_0_Relu_ts[1,8]
   + A[1,1] * Conv_1_aggregated_features[1,8]
   - A[1,2] * Conv_0_Relu_ts[2,8] - A[1,3] * Conv_0_Relu_ts[3,8]
   - A[1,4] * Conv_0_Relu_ts[4,8] - A[1,5] * Conv_0_Relu_ts[5,8]
   - A[1,6] * Conv_0_Relu_ts[6,8] - A[1,7] * Conv_0_Relu_ts[7,8]
   + A[2,1] * Conv_1_aggregated_features[1,8]
   + A[3,1] * Conv_1_aggregated_features[1,8]
   + A[4,1] * Conv_1_aggregated_features[1,8]
   + A[5,1] * Conv_1_aggregated_features[1,8]
   + A[6,1] * Conv_1_aggregated_features[1,8]
   + A[7,1] * Conv_1_aggregated_features[1,8] ] = 0
 Conv_1_averages_constraint[1,9]: [
   A[0,1] * Conv_1_aggregated_features[1,9] - A[1,0] * Conv_0_Relu_ts[0,9]
   - A[1,1] * Conv_0_Relu_ts[1,9]
   + A[1,1] * Conv_1_aggregated_features[1,9]
   - A[1,2] * Conv_0_Relu_ts[2,9] - A[1,3] * Conv_0_Relu_ts[3,9]
   - A[1,4] * Conv_0_Relu_ts[4,9] - A[1,5] * Conv_0_Relu_ts[5,9]
   - A[1,6] * Conv_0_Relu_ts[6,9] - A[1,7] * Conv_0_Relu_ts[7,9]
   + A[2,1] * Conv_1_aggregated_features[1,9]
   + A[3,1] * Conv_1_aggregated_features[1,9]
   + A[4,1] * Conv_1_aggregated_features[1,9]
   + A[5,1] * Conv_1_aggregated_features[1,9]
   + A[6,1] * Conv_1_aggregated_features[1,9]
   + A[7,1] * Conv_1_aggregated_features[1,9] ] = 0
 Conv_1_averages_constraint[1,10]: [
   A[0,1] * Conv_1_aggregated_features[1,10]
   - A[1,0] * Conv_0_Relu_ts[0,10] - A[1,1] * Conv_0_Relu_ts[1,10]
   + A[1,1] * Conv_1_aggregated_features[1,10]
   - A[1,2] * Conv_0_Relu_ts[2,10] - A[1,3] * Conv_0_Relu_ts[3,10]
   - A[1,4] * Conv_0_Relu_ts[4,10] - A[1,5] * Conv_0_Relu_ts[5,10]
   - A[1,6] * Conv_0_Relu_ts[6,10] - A[1,7] * Conv_0_Relu_ts[7,10]
   + A[2,1] * Conv_1_aggregated_features[1,10]
   + A[3,1] * Conv_1_aggregated_features[1,10]
   + A[4,1] * Conv_1_aggregated_features[1,10]
   + A[5,1] * Conv_1_aggregated_features[1,10]
   + A[6,1] * Conv_1_aggregated_features[1,10]
   + A[7,1] * Conv_1_aggregated_features[1,10] ] = 0
 Conv_1_averages_constraint[1,11]: [
   A[0,1] * Conv_1_aggregated_features[1,11]
   - A[1,0] * Conv_0_Relu_ts[0,11] - A[1,1] * Conv_0_Relu_ts[1,11]
   + A[1,1] * Conv_1_aggregated_features[1,11]
   - A[1,2] * Conv_0_Relu_ts[2,11] - A[1,3] * Conv_0_Relu_ts[3,11]
   - A[1,4] * Conv_0_Relu_ts[4,11] - A[1,5] * Conv_0_Relu_ts[5,11]
   - A[1,6] * Conv_0_Relu_ts[6,11] - A[1,7] * Conv_0_Relu_ts[7,11]
   + A[2,1] * Conv_1_aggregated_features[1,11]
   + A[3,1] * Conv_1_aggregated_features[1,11]
   + A[4,1] * Conv_1_aggregated_features[1,11]
   + A[5,1] * Conv_1_aggregated_features[1,11]
   + A[6,1] * Conv_1_aggregated_features[1,11]
   + A[7,1] * Conv_1_aggregated_features[1,11] ] = 0
 Conv_1_averages_constraint[1,12]: [
   A[0,1] * Conv_1_aggregated_features[1,12]
   - A[1,0] * Conv_0_Relu_ts[0,12] - A[1,1] * Conv_0_Relu_ts[1,12]
   + A[1,1] * Conv_1_aggregated_features[1,12]
   - A[1,2] * Conv_0_Relu_ts[2,12] - A[1,3] * Conv_0_Relu_ts[3,12]
   - A[1,4] * Conv_0_Relu_ts[4,12] - A[1,5] * Conv_0_Relu_ts[5,12]
   - A[1,6] * Conv_0_Relu_ts[6,12] - A[1,7] * Conv_0_Relu_ts[7,12]
   + A[2,1] * Conv_1_aggregated_features[1,12]
   + A[3,1] * Conv_1_aggregated_features[1,12]
   + A[4,1] * Conv_1_aggregated_features[1,12]
   + A[5,1] * Conv_1_aggregated_features[1,12]
   + A[6,1] * Conv_1_aggregated_features[1,12]
   + A[7,1] * Conv_1_aggregated_features[1,12] ] = 0
 Conv_1_averages_constraint[1,13]: [
   A[0,1] * Conv_1_aggregated_features[1,13]
   - A[1,0] * Conv_0_Relu_ts[0,13] - A[1,1] * Conv_0_Relu_ts[1,13]
   + A[1,1] * Conv_1_aggregated_features[1,13]
   - A[1,2] * Conv_0_Relu_ts[2,13] - A[1,3] * Conv_0_Relu_ts[3,13]
   - A[1,4] * Conv_0_Relu_ts[4,13] - A[1,5] * Conv_0_Relu_ts[5,13]
   - A[1,6] * Conv_0_Relu_ts[6,13] - A[1,7] * Conv_0_Relu_ts[7,13]
   + A[2,1] * Conv_1_aggregated_features[1,13]
   + A[3,1] * Conv_1_aggregated_features[1,13]
   + A[4,1] * Conv_1_aggregated_features[1,13]
   + A[5,1] * Conv_1_aggregated_features[1,13]
   + A[6,1] * Conv_1_aggregated_features[1,13]
   + A[7,1] * Conv_1_aggregated_features[1,13] ] = 0
 Conv_1_averages_constraint[1,14]: [
   A[0,1] * Conv_1_aggregated_features[1,14]
   - A[1,0] * Conv_0_Relu_ts[0,14] - A[1,1] * Conv_0_Relu_ts[1,14]
   + A[1,1] * Conv_1_aggregated_features[1,14]
   - A[1,2] * Conv_0_Relu_ts[2,14] - A[1,3] * Conv_0_Relu_ts[3,14]
   - A[1,4] * Conv_0_Relu_ts[4,14] - A[1,5] * Conv_0_Relu_ts[5,14]
   - A[1,6] * Conv_0_Relu_ts[6,14] - A[1,7] * Conv_0_Relu_ts[7,14]
   + A[2,1] * Conv_1_aggregated_features[1,14]
   + A[3,1] * Conv_1_aggregated_features[1,14]
   + A[4,1] * Conv_1_aggregated_features[1,14]
   + A[5,1] * Conv_1_aggregated_features[1,14]
   + A[6,1] * Conv_1_aggregated_features[1,14]
   + A[7,1] * Conv_1_aggregated_features[1,14] ] = 0
 Conv_1_averages_constraint[1,15]: [
   A[0,1] * Conv_1_aggregated_features[1,15]
   - A[1,0] * Conv_0_Relu_ts[0,15] - A[1,1] * Conv_0_Relu_ts[1,15]
   + A[1,1] * Conv_1_aggregated_features[1,15]
   - A[1,2] * Conv_0_Relu_ts[2,15] - A[1,3] * Conv_0_Relu_ts[3,15]
   - A[1,4] * Conv_0_Relu_ts[4,15] - A[1,5] * Conv_0_Relu_ts[5,15]
   - A[1,6] * Conv_0_Relu_ts[6,15] - A[1,7] * Conv_0_Relu_ts[7,15]
   + A[2,1] * Conv_1_aggregated_features[1,15]
   + A[3,1] * Conv_1_aggregated_features[1,15]
   + A[4,1] * Conv_1_aggregated_features[1,15]
   + A[5,1] * Conv_1_aggregated_features[1,15]
   + A[6,1] * Conv_1_aggregated_features[1,15]
   + A[7,1] * Conv_1_aggregated_features[1,15] ] = 0
 Conv_1_averages_constraint[2,0]: [
   A[0,2] * Conv_1_aggregated_features[2,0]
   + A[1,2] * Conv_1_aggregated_features[2,0]
   - A[2,0] * Conv_0_Relu_ts[0,0] - A[2,1] * Conv_0_Relu_ts[1,0]
   - A[2,2] * Conv_0_Relu_ts[2,0]
   + A[2,2] * Conv_1_aggregated_features[2,0]
   - A[2,3] * Conv_0_Relu_ts[3,0] - A[2,4] * Conv_0_Relu_ts[4,0]
   - A[2,5] * Conv_0_Relu_ts[5,0] - A[2,6] * Conv_0_Relu_ts[6,0]
   - A[2,7] * Conv_0_Relu_ts[7,0]
   + A[3,2] * Conv_1_aggregated_features[2,0]
   + A[4,2] * Conv_1_aggregated_features[2,0]
   + A[5,2] * Conv_1_aggregated_features[2,0]
   + A[6,2] * Conv_1_aggregated_features[2,0]
   + A[7,2] * Conv_1_aggregated_features[2,0] ] = 0
 Conv_1_averages_constraint[2,1]: [
   A[0,2] * Conv_1_aggregated_features[2,1]
   + A[1,2] * Conv_1_aggregated_features[2,1]
   - A[2,0] * Conv_0_Relu_ts[0,1] - A[2,1] * Conv_0_Relu_ts[1,1]
   - A[2,2] * Conv_0_Relu_ts[2,1]
   + A[2,2] * Conv_1_aggregated_features[2,1]
   - A[2,3] * Conv_0_Relu_ts[3,1] - A[2,4] * Conv_0_Relu_ts[4,1]
   - A[2,5] * Conv_0_Relu_ts[5,1] - A[2,6] * Conv_0_Relu_ts[6,1]
   - A[2,7] * Conv_0_Relu_ts[7,1]
   + A[3,2] * Conv_1_aggregated_features[2,1]
   + A[4,2] * Conv_1_aggregated_features[2,1]
   + A[5,2] * Conv_1_aggregated_features[2,1]
   + A[6,2] * Conv_1_aggregated_features[2,1]
   + A[7,2] * Conv_1_aggregated_features[2,1] ] = 0
 Conv_1_averages_constraint[2,2]: [
   A[0,2] * Conv_1_aggregated_features[2,2]
   + A[1,2] * Conv_1_aggregated_features[2,2]
   - A[2,0] * Conv_0_Relu_ts[0,2] - A[2,1] * Conv_0_Relu_ts[1,2]
   - A[2,2] * Conv_0_Relu_ts[2,2]
   + A[2,2] * Conv_1_aggregated_features[2,2]
   - A[2,3] * Conv_0_Relu_ts[3,2] - A[2,4] * Conv_0_Relu_ts[4,2]
   - A[2,5] * Conv_0_Relu_ts[5,2] - A[2,6] * Conv_0_Relu_ts[6,2]
   - A[2,7] * Conv_0_Relu_ts[7,2]
   + A[3,2] * Conv_1_aggregated_features[2,2]
   + A[4,2] * Conv_1_aggregated_features[2,2]
   + A[5,2] * Conv_1_aggregated_features[2,2]
   + A[6,2] * Conv_1_aggregated_features[2,2]
   + A[7,2] * Conv_1_aggregated_features[2,2] ] = 0
 Conv_1_averages_constraint[2,3]: [
   A[0,2] * Conv_1_aggregated_features[2,3]
   + A[1,2] * Conv_1_aggregated_features[2,3]
   - A[2,0] * Conv_0_Relu_ts[0,3] - A[2,1] * Conv_0_Relu_ts[1,3]
   - A[2,2] * Conv_0_Relu_ts[2,3]
   + A[2,2] * Conv_1_aggregated_features[2,3]
   - A[2,3] * Conv_0_Relu_ts[3,3] - A[2,4] * Conv_0_Relu_ts[4,3]
   - A[2,5] * Conv_0_Relu_ts[5,3] - A[2,6] * Conv_0_Relu_ts[6,3]
   - A[2,7] * Conv_0_Relu_ts[7,3]
   + A[3,2] * Conv_1_aggregated_features[2,3]
   + A[4,2] * Conv_1_aggregated_features[2,3]
   + A[5,2] * Conv_1_aggregated_features[2,3]
   + A[6,2] * Conv_1_aggregated_features[2,3]
   + A[7,2] * Conv_1_aggregated_features[2,3] ] = 0
 Conv_1_averages_constraint[2,4]: [
   A[0,2] * Conv_1_aggregated_features[2,4]
   + A[1,2] * Conv_1_aggregated_features[2,4]
   - A[2,0] * Conv_0_Relu_ts[0,4] - A[2,1] * Conv_0_Relu_ts[1,4]
   - A[2,2] * Conv_0_Relu_ts[2,4]
   + A[2,2] * Conv_1_aggregated_features[2,4]
   - A[2,3] * Conv_0_Relu_ts[3,4] - A[2,4] * Conv_0_Relu_ts[4,4]
   - A[2,5] * Conv_0_Relu_ts[5,4] - A[2,6] * Conv_0_Relu_ts[6,4]
   - A[2,7] * Conv_0_Relu_ts[7,4]
   + A[3,2] * Conv_1_aggregated_features[2,4]
   + A[4,2] * Conv_1_aggregated_features[2,4]
   + A[5,2] * Conv_1_aggregated_features[2,4]
   + A[6,2] * Conv_1_aggregated_features[2,4]
   + A[7,2] * Conv_1_aggregated_features[2,4] ] = 0
 Conv_1_averages_constraint[2,5]: [
   A[0,2] * Conv_1_aggregated_features[2,5]
   + A[1,2] * Conv_1_aggregated_features[2,5]
   - A[2,0] * Conv_0_Relu_ts[0,5] - A[2,1] * Conv_0_Relu_ts[1,5]
   - A[2,2] * Conv_0_Relu_ts[2,5]
   + A[2,2] * Conv_1_aggregated_features[2,5]
   - A[2,3] * Conv_0_Relu_ts[3,5] - A[2,4] * Conv_0_Relu_ts[4,5]
   - A[2,5] * Conv_0_Relu_ts[5,5] - A[2,6] * Conv_0_Relu_ts[6,5]
   - A[2,7] * Conv_0_Relu_ts[7,5]
   + A[3,2] * Conv_1_aggregated_features[2,5]
   + A[4,2] * Conv_1_aggregated_features[2,5]
   + A[5,2] * Conv_1_aggregated_features[2,5]
   + A[6,2] * Conv_1_aggregated_features[2,5]
   + A[7,2] * Conv_1_aggregated_features[2,5] ] = 0
 Conv_1_averages_constraint[2,6]: [
   A[0,2] * Conv_1_aggregated_features[2,6]
   + A[1,2] * Conv_1_aggregated_features[2,6]
   - A[2,0] * Conv_0_Relu_ts[0,6] - A[2,1] * Conv_0_Relu_ts[1,6]
   - A[2,2] * Conv_0_Relu_ts[2,6]
   + A[2,2] * Conv_1_aggregated_features[2,6]
   - A[2,3] * Conv_0_Relu_ts[3,6] - A[2,4] * Conv_0_Relu_ts[4,6]
   - A[2,5] * Conv_0_Relu_ts[5,6] - A[2,6] * Conv_0_Relu_ts[6,6]
   - A[2,7] * Conv_0_Relu_ts[7,6]
   + A[3,2] * Conv_1_aggregated_features[2,6]
   + A[4,2] * Conv_1_aggregated_features[2,6]
   + A[5,2] * Conv_1_aggregated_features[2,6]
   + A[6,2] * Conv_1_aggregated_features[2,6]
   + A[7,2] * Conv_1_aggregated_features[2,6] ] = 0
 Conv_1_averages_constraint[2,7]: [
   A[0,2] * Conv_1_aggregated_features[2,7]
   + A[1,2] * Conv_1_aggregated_features[2,7]
   - A[2,0] * Conv_0_Relu_ts[0,7] - A[2,1] * Conv_0_Relu_ts[1,7]
   - A[2,2] * Conv_0_Relu_ts[2,7]
   + A[2,2] * Conv_1_aggregated_features[2,7]
   - A[2,3] * Conv_0_Relu_ts[3,7] - A[2,4] * Conv_0_Relu_ts[4,7]
   - A[2,5] * Conv_0_Relu_ts[5,7] - A[2,6] * Conv_0_Relu_ts[6,7]
   - A[2,7] * Conv_0_Relu_ts[7,7]
   + A[3,2] * Conv_1_aggregated_features[2,7]
   + A[4,2] * Conv_1_aggregated_features[2,7]
   + A[5,2] * Conv_1_aggregated_features[2,7]
   + A[6,2] * Conv_1_aggregated_features[2,7]
   + A[7,2] * Conv_1_aggregated_features[2,7] ] = 0
 Conv_1_averages_constraint[2,8]: [
   A[0,2] * Conv_1_aggregated_features[2,8]
   + A[1,2] * Conv_1_aggregated_features[2,8]
   - A[2,0] * Conv_0_Relu_ts[0,8] - A[2,1] * Conv_0_Relu_ts[1,8]
   - A[2,2] * Conv_0_Relu_ts[2,8]
   + A[2,2] * Conv_1_aggregated_features[2,8]
   - A[2,3] * Conv_0_Relu_ts[3,8] - A[2,4] * Conv_0_Relu_ts[4,8]
   - A[2,5] * Conv_0_Relu_ts[5,8] - A[2,6] * Conv_0_Relu_ts[6,8]
   - A[2,7] * Conv_0_Relu_ts[7,8]
   + A[3,2] * Conv_1_aggregated_features[2,8]
   + A[4,2] * Conv_1_aggregated_features[2,8]
   + A[5,2] * Conv_1_aggregated_features[2,8]
   + A[6,2] * Conv_1_aggregated_features[2,8]
   + A[7,2] * Conv_1_aggregated_features[2,8] ] = 0
 Conv_1_averages_constraint[2,9]: [
   A[0,2] * Conv_1_aggregated_features[2,9]
   + A[1,2] * Conv_1_aggregated_features[2,9]
   - A[2,0] * Conv_0_Relu_ts[0,9] - A[2,1] * Conv_0_Relu_ts[1,9]
   - A[2,2] * Conv_0_Relu_ts[2,9]
   + A[2,2] * Conv_1_aggregated_features[2,9]
   - A[2,3] * Conv_0_Relu_ts[3,9] - A[2,4] * Conv_0_Relu_ts[4,9]
   - A[2,5] * Conv_0_Relu_ts[5,9] - A[2,6] * Conv_0_Relu_ts[6,9]
   - A[2,7] * Conv_0_Relu_ts[7,9]
   + A[3,2] * Conv_1_aggregated_features[2,9]
   + A[4,2] * Conv_1_aggregated_features[2,9]
   + A[5,2] * Conv_1_aggregated_features[2,9]
   + A[6,2] * Conv_1_aggregated_features[2,9]
   + A[7,2] * Conv_1_aggregated_features[2,9] ] = 0
 Conv_1_averages_constraint[2,10]: [
   A[0,2] * Conv_1_aggregated_features[2,10]
   + A[1,2] * Conv_1_aggregated_features[2,10]
   - A[2,0] * Conv_0_Relu_ts[0,10] - A[2,1] * Conv_0_Relu_ts[1,10]
   - A[2,2] * Conv_0_Relu_ts[2,10]
   + A[2,2] * Conv_1_aggregated_features[2,10]
   - A[2,3] * Conv_0_Relu_ts[3,10] - A[2,4] * Conv_0_Relu_ts[4,10]
   - A[2,5] * Conv_0_Relu_ts[5,10] - A[2,6] * Conv_0_Relu_ts[6,10]
   - A[2,7] * Conv_0_Relu_ts[7,10]
   + A[3,2] * Conv_1_aggregated_features[2,10]
   + A[4,2] * Conv_1_aggregated_features[2,10]
   + A[5,2] * Conv_1_aggregated_features[2,10]
   + A[6,2] * Conv_1_aggregated_features[2,10]
   + A[7,2] * Conv_1_aggregated_features[2,10] ] = 0
 Conv_1_averages_constraint[2,11]: [
   A[0,2] * Conv_1_aggregated_features[2,11]
   + A[1,2] * Conv_1_aggregated_features[2,11]
   - A[2,0] * Conv_0_Relu_ts[0,11] - A[2,1] * Conv_0_Relu_ts[1,11]
   - A[2,2] * Conv_0_Relu_ts[2,11]
   + A[2,2] * Conv_1_aggregated_features[2,11]
   - A[2,3] * Conv_0_Relu_ts[3,11] - A[2,4] * Conv_0_Relu_ts[4,11]
   - A[2,5] * Conv_0_Relu_ts[5,11] - A[2,6] * Conv_0_Relu_ts[6,11]
   - A[2,7] * Conv_0_Relu_ts[7,11]
   + A[3,2] * Conv_1_aggregated_features[2,11]
   + A[4,2] * Conv_1_aggregated_features[2,11]
   + A[5,2] * Conv_1_aggregated_features[2,11]
   + A[6,2] * Conv_1_aggregated_features[2,11]
   + A[7,2] * Conv_1_aggregated_features[2,11] ] = 0
 Conv_1_averages_constraint[2,12]: [
   A[0,2] * Conv_1_aggregated_features[2,12]
   + A[1,2] * Conv_1_aggregated_features[2,12]
   - A[2,0] * Conv_0_Relu_ts[0,12] - A[2,1] * Conv_0_Relu_ts[1,12]
   - A[2,2] * Conv_0_Relu_ts[2,12]
   + A[2,2] * Conv_1_aggregated_features[2,12]
   - A[2,3] * Conv_0_Relu_ts[3,12] - A[2,4] * Conv_0_Relu_ts[4,12]
   - A[2,5] * Conv_0_Relu_ts[5,12] - A[2,6] * Conv_0_Relu_ts[6,12]
   - A[2,7] * Conv_0_Relu_ts[7,12]
   + A[3,2] * Conv_1_aggregated_features[2,12]
   + A[4,2] * Conv_1_aggregated_features[2,12]
   + A[5,2] * Conv_1_aggregated_features[2,12]
   + A[6,2] * Conv_1_aggregated_features[2,12]
   + A[7,2] * Conv_1_aggregated_features[2,12] ] = 0
 Conv_1_averages_constraint[2,13]: [
   A[0,2] * Conv_1_aggregated_features[2,13]
   + A[1,2] * Conv_1_aggregated_features[2,13]
   - A[2,0] * Conv_0_Relu_ts[0,13] - A[2,1] * Conv_0_Relu_ts[1,13]
   - A[2,2] * Conv_0_Relu_ts[2,13]
   + A[2,2] * Conv_1_aggregated_features[2,13]
   - A[2,3] * Conv_0_Relu_ts[3,13] - A[2,4] * Conv_0_Relu_ts[4,13]
   - A[2,5] * Conv_0_Relu_ts[5,13] - A[2,6] * Conv_0_Relu_ts[6,13]
   - A[2,7] * Conv_0_Relu_ts[7,13]
   + A[3,2] * Conv_1_aggregated_features[2,13]
   + A[4,2] * Conv_1_aggregated_features[2,13]
   + A[5,2] * Conv_1_aggregated_features[2,13]
   + A[6,2] * Conv_1_aggregated_features[2,13]
   + A[7,2] * Conv_1_aggregated_features[2,13] ] = 0
 Conv_1_averages_constraint[2,14]: [
   A[0,2] * Conv_1_aggregated_features[2,14]
   + A[1,2] * Conv_1_aggregated_features[2,14]
   - A[2,0] * Conv_0_Relu_ts[0,14] - A[2,1] * Conv_0_Relu_ts[1,14]
   - A[2,2] * Conv_0_Relu_ts[2,14]
   + A[2,2] * Conv_1_aggregated_features[2,14]
   - A[2,3] * Conv_0_Relu_ts[3,14] - A[2,4] * Conv_0_Relu_ts[4,14]
   - A[2,5] * Conv_0_Relu_ts[5,14] - A[2,6] * Conv_0_Relu_ts[6,14]
   - A[2,7] * Conv_0_Relu_ts[7,14]
   + A[3,2] * Conv_1_aggregated_features[2,14]
   + A[4,2] * Conv_1_aggregated_features[2,14]
   + A[5,2] * Conv_1_aggregated_features[2,14]
   + A[6,2] * Conv_1_aggregated_features[2,14]
   + A[7,2] * Conv_1_aggregated_features[2,14] ] = 0
 Conv_1_averages_constraint[2,15]: [
   A[0,2] * Conv_1_aggregated_features[2,15]
   + A[1,2] * Conv_1_aggregated_features[2,15]
   - A[2,0] * Conv_0_Relu_ts[0,15] - A[2,1] * Conv_0_Relu_ts[1,15]
   - A[2,2] * Conv_0_Relu_ts[2,15]
   + A[2,2] * Conv_1_aggregated_features[2,15]
   - A[2,3] * Conv_0_Relu_ts[3,15] - A[2,4] * Conv_0_Relu_ts[4,15]
   - A[2,5] * Conv_0_Relu_ts[5,15] - A[2,6] * Conv_0_Relu_ts[6,15]
   - A[2,7] * Conv_0_Relu_ts[7,15]
   + A[3,2] * Conv_1_aggregated_features[2,15]
   + A[4,2] * Conv_1_aggregated_features[2,15]
   + A[5,2] * Conv_1_aggregated_features[2,15]
   + A[6,2] * Conv_1_aggregated_features[2,15]
   + A[7,2] * Conv_1_aggregated_features[2,15] ] = 0
 Conv_1_averages_constraint[3,0]: [
   A[0,3] * Conv_1_aggregated_features[3,0]
   + A[1,3] * Conv_1_aggregated_features[3,0]
   + A[2,3] * Conv_1_aggregated_features[3,0]
   - A[3,0] * Conv_0_Relu_ts[0,0] - A[3,1] * Conv_0_Relu_ts[1,0]
   - A[3,2] * Conv_0_Relu_ts[2,0] - A[3,3] * Conv_0_Relu_ts[3,0]
   + A[3,3] * Conv_1_aggregated_features[3,0]
   - A[3,4] * Conv_0_Relu_ts[4,0] - A[3,5] * Conv_0_Relu_ts[5,0]
   - A[3,6] * Conv_0_Relu_ts[6,0] - A[3,7] * Conv_0_Relu_ts[7,0]
   + A[4,3] * Conv_1_aggregated_features[3,0]
   + A[5,3] * Conv_1_aggregated_features[3,0]
   + A[6,3] * Conv_1_aggregated_features[3,0]
   + A[7,3] * Conv_1_aggregated_features[3,0] ] = 0
 Conv_1_averages_constraint[3,1]: [
   A[0,3] * Conv_1_aggregated_features[3,1]
   + A[1,3] * Conv_1_aggregated_features[3,1]
   + A[2,3] * Conv_1_aggregated_features[3,1]
   - A[3,0] * Conv_0_Relu_ts[0,1] - A[3,1] * Conv_0_Relu_ts[1,1]
   - A[3,2] * Conv_0_Relu_ts[2,1] - A[3,3] * Conv_0_Relu_ts[3,1]
   + A[3,3] * Conv_1_aggregated_features[3,1]
   - A[3,4] * Conv_0_Relu_ts[4,1] - A[3,5] * Conv_0_Relu_ts[5,1]
   - A[3,6] * Conv_0_Relu_ts[6,1] - A[3,7] * Conv_0_Relu_ts[7,1]
   + A[4,3] * Conv_1_aggregated_features[3,1]
   + A[5,3] * Conv_1_aggregated_features[3,1]
   + A[6,3] * Conv_1_aggregated_features[3,1]
   + A[7,3] * Conv_1_aggregated_features[3,1] ] = 0
 Conv_1_averages_constraint[3,2]: [
   A[0,3] * Conv_1_aggregated_features[3,2]
   + A[1,3] * Conv_1_aggregated_features[3,2]
   + A[2,3] * Conv_1_aggregated_features[3,2]
   - A[3,0] * Conv_0_Relu_ts[0,2] - A[3,1] * Conv_0_Relu_ts[1,2]
   - A[3,2] * Conv_0_Relu_ts[2,2] - A[3,3] * Conv_0_Relu_ts[3,2]
   + A[3,3] * Conv_1_aggregated_features[3,2]
   - A[3,4] * Conv_0_Relu_ts[4,2] - A[3,5] * Conv_0_Relu_ts[5,2]
   - A[3,6] * Conv_0_Relu_ts[6,2] - A[3,7] * Conv_0_Relu_ts[7,2]
   + A[4,3] * Conv_1_aggregated_features[3,2]
   + A[5,3] * Conv_1_aggregated_features[3,2]
   + A[6,3] * Conv_1_aggregated_features[3,2]
   + A[7,3] * Conv_1_aggregated_features[3,2] ] = 0
 Conv_1_averages_constraint[3,3]: [
   A[0,3] * Conv_1_aggregated_features[3,3]
   + A[1,3] * Conv_1_aggregated_features[3,3]
   + A[2,3] * Conv_1_aggregated_features[3,3]
   - A[3,0] * Conv_0_Relu_ts[0,3] - A[3,1] * Conv_0_Relu_ts[1,3]
   - A[3,2] * Conv_0_Relu_ts[2,3] - A[3,3] * Conv_0_Relu_ts[3,3]
   + A[3,3] * Conv_1_aggregated_features[3,3]
   - A[3,4] * Conv_0_Relu_ts[4,3] - A[3,5] * Conv_0_Relu_ts[5,3]
   - A[3,6] * Conv_0_Relu_ts[6,3] - A[3,7] * Conv_0_Relu_ts[7,3]
   + A[4,3] * Conv_1_aggregated_features[3,3]
   + A[5,3] * Conv_1_aggregated_features[3,3]
   + A[6,3] * Conv_1_aggregated_features[3,3]
   + A[7,3] * Conv_1_aggregated_features[3,3] ] = 0
 Conv_1_averages_constraint[3,4]: [
   A[0,3] * Conv_1_aggregated_features[3,4]
   + A[1,3] * Conv_1_aggregated_features[3,4]
   + A[2,3] * Conv_1_aggregated_features[3,4]
   - A[3,0] * Conv_0_Relu_ts[0,4] - A[3,1] * Conv_0_Relu_ts[1,4]
   - A[3,2] * Conv_0_Relu_ts[2,4] - A[3,3] * Conv_0_Relu_ts[3,4]
   + A[3,3] * Conv_1_aggregated_features[3,4]
   - A[3,4] * Conv_0_Relu_ts[4,4] - A[3,5] * Conv_0_Relu_ts[5,4]
   - A[3,6] * Conv_0_Relu_ts[6,4] - A[3,7] * Conv_0_Relu_ts[7,4]
   + A[4,3] * Conv_1_aggregated_features[3,4]
   + A[5,3] * Conv_1_aggregated_features[3,4]
   + A[6,3] * Conv_1_aggregated_features[3,4]
   + A[7,3] * Conv_1_aggregated_features[3,4] ] = 0
 Conv_1_averages_constraint[3,5]: [
   A[0,3] * Conv_1_aggregated_features[3,5]
   + A[1,3] * Conv_1_aggregated_features[3,5]
   + A[2,3] * Conv_1_aggregated_features[3,5]
   - A[3,0] * Conv_0_Relu_ts[0,5] - A[3,1] * Conv_0_Relu_ts[1,5]
   - A[3,2] * Conv_0_Relu_ts[2,5] - A[3,3] * Conv_0_Relu_ts[3,5]
   + A[3,3] * Conv_1_aggregated_features[3,5]
   - A[3,4] * Conv_0_Relu_ts[4,5] - A[3,5] * Conv_0_Relu_ts[5,5]
   - A[3,6] * Conv_0_Relu_ts[6,5] - A[3,7] * Conv_0_Relu_ts[7,5]
   + A[4,3] * Conv_1_aggregated_features[3,5]
   + A[5,3] * Conv_1_aggregated_features[3,5]
   + A[6,3] * Conv_1_aggregated_features[3,5]
   + A[7,3] * Conv_1_aggregated_features[3,5] ] = 0
 Conv_1_averages_constraint[3,6]: [
   A[0,3] * Conv_1_aggregated_features[3,6]
   + A[1,3] * Conv_1_aggregated_features[3,6]
   + A[2,3] * Conv_1_aggregated_features[3,6]
   - A[3,0] * Conv_0_Relu_ts[0,6] - A[3,1] * Conv_0_Relu_ts[1,6]
   - A[3,2] * Conv_0_Relu_ts[2,6] - A[3,3] * Conv_0_Relu_ts[3,6]
   + A[3,3] * Conv_1_aggregated_features[3,6]
   - A[3,4] * Conv_0_Relu_ts[4,6] - A[3,5] * Conv_0_Relu_ts[5,6]
   - A[3,6] * Conv_0_Relu_ts[6,6] - A[3,7] * Conv_0_Relu_ts[7,6]
   + A[4,3] * Conv_1_aggregated_features[3,6]
   + A[5,3] * Conv_1_aggregated_features[3,6]
   + A[6,3] * Conv_1_aggregated_features[3,6]
   + A[7,3] * Conv_1_aggregated_features[3,6] ] = 0
 Conv_1_averages_constraint[3,7]: [
   A[0,3] * Conv_1_aggregated_features[3,7]
   + A[1,3] * Conv_1_aggregated_features[3,7]
   + A[2,3] * Conv_1_aggregated_features[3,7]
   - A[3,0] * Conv_0_Relu_ts[0,7] - A[3,1] * Conv_0_Relu_ts[1,7]
   - A[3,2] * Conv_0_Relu_ts[2,7] - A[3,3] * Conv_0_Relu_ts[3,7]
   + A[3,3] * Conv_1_aggregated_features[3,7]
   - A[3,4] * Conv_0_Relu_ts[4,7] - A[3,5] * Conv_0_Relu_ts[5,7]
   - A[3,6] * Conv_0_Relu_ts[6,7] - A[3,7] * Conv_0_Relu_ts[7,7]
   + A[4,3] * Conv_1_aggregated_features[3,7]
   + A[5,3] * Conv_1_aggregated_features[3,7]
   + A[6,3] * Conv_1_aggregated_features[3,7]
   + A[7,3] * Conv_1_aggregated_features[3,7] ] = 0
 Conv_1_averages_constraint[3,8]: [
   A[0,3] * Conv_1_aggregated_features[3,8]
   + A[1,3] * Conv_1_aggregated_features[3,8]
   + A[2,3] * Conv_1_aggregated_features[3,8]
   - A[3,0] * Conv_0_Relu_ts[0,8] - A[3,1] * Conv_0_Relu_ts[1,8]
   - A[3,2] * Conv_0_Relu_ts[2,8] - A[3,3] * Conv_0_Relu_ts[3,8]
   + A[3,3] * Conv_1_aggregated_features[3,8]
   - A[3,4] * Conv_0_Relu_ts[4,8] - A[3,5] * Conv_0_Relu_ts[5,8]
   - A[3,6] * Conv_0_Relu_ts[6,8] - A[3,7] * Conv_0_Relu_ts[7,8]
   + A[4,3] * Conv_1_aggregated_features[3,8]
   + A[5,3] * Conv_1_aggregated_features[3,8]
   + A[6,3] * Conv_1_aggregated_features[3,8]
   + A[7,3] * Conv_1_aggregated_features[3,8] ] = 0
 Conv_1_averages_constraint[3,9]: [
   A[0,3] * Conv_1_aggregated_features[3,9]
   + A[1,3] * Conv_1_aggregated_features[3,9]
   + A[2,3] * Conv_1_aggregated_features[3,9]
   - A[3,0] * Conv_0_Relu_ts[0,9] - A[3,1] * Conv_0_Relu_ts[1,9]
   - A[3,2] * Conv_0_Relu_ts[2,9] - A[3,3] * Conv_0_Relu_ts[3,9]
   + A[3,3] * Conv_1_aggregated_features[3,9]
   - A[3,4] * Conv_0_Relu_ts[4,9] - A[3,5] * Conv_0_Relu_ts[5,9]
   - A[3,6] * Conv_0_Relu_ts[6,9] - A[3,7] * Conv_0_Relu_ts[7,9]
   + A[4,3] * Conv_1_aggregated_features[3,9]
   + A[5,3] * Conv_1_aggregated_features[3,9]
   + A[6,3] * Conv_1_aggregated_features[3,9]
   + A[7,3] * Conv_1_aggregated_features[3,9] ] = 0
 Conv_1_averages_constraint[3,10]: [
   A[0,3] * Conv_1_aggregated_features[3,10]
   + A[1,3] * Conv_1_aggregated_features[3,10]
   + A[2,3] * Conv_1_aggregated_features[3,10]
   - A[3,0] * Conv_0_Relu_ts[0,10] - A[3,1] * Conv_0_Relu_ts[1,10]
   - A[3,2] * Conv_0_Relu_ts[2,10] - A[3,3] * Conv_0_Relu_ts[3,10]
   + A[3,3] * Conv_1_aggregated_features[3,10]
   - A[3,4] * Conv_0_Relu_ts[4,10] - A[3,5] * Conv_0_Relu_ts[5,10]
   - A[3,6] * Conv_0_Relu_ts[6,10] - A[3,7] * Conv_0_Relu_ts[7,10]
   + A[4,3] * Conv_1_aggregated_features[3,10]
   + A[5,3] * Conv_1_aggregated_features[3,10]
   + A[6,3] * Conv_1_aggregated_features[3,10]
   + A[7,3] * Conv_1_aggregated_features[3,10] ] = 0
 Conv_1_averages_constraint[3,11]: [
   A[0,3] * Conv_1_aggregated_features[3,11]
   + A[1,3] * Conv_1_aggregated_features[3,11]
   + A[2,3] * Conv_1_aggregated_features[3,11]
   - A[3,0] * Conv_0_Relu_ts[0,11] - A[3,1] * Conv_0_Relu_ts[1,11]
   - A[3,2] * Conv_0_Relu_ts[2,11] - A[3,3] * Conv_0_Relu_ts[3,11]
   + A[3,3] * Conv_1_aggregated_features[3,11]
   - A[3,4] * Conv_0_Relu_ts[4,11] - A[3,5] * Conv_0_Relu_ts[5,11]
   - A[3,6] * Conv_0_Relu_ts[6,11] - A[3,7] * Conv_0_Relu_ts[7,11]
   + A[4,3] * Conv_1_aggregated_features[3,11]
   + A[5,3] * Conv_1_aggregated_features[3,11]
   + A[6,3] * Conv_1_aggregated_features[3,11]
   + A[7,3] * Conv_1_aggregated_features[3,11] ] = 0
 Conv_1_averages_constraint[3,12]: [
   A[0,3] * Conv_1_aggregated_features[3,12]
   + A[1,3] * Conv_1_aggregated_features[3,12]
   + A[2,3] * Conv_1_aggregated_features[3,12]
   - A[3,0] * Conv_0_Relu_ts[0,12] - A[3,1] * Conv_0_Relu_ts[1,12]
   - A[3,2] * Conv_0_Relu_ts[2,12] - A[3,3] * Conv_0_Relu_ts[3,12]
   + A[3,3] * Conv_1_aggregated_features[3,12]
   - A[3,4] * Conv_0_Relu_ts[4,12] - A[3,5] * Conv_0_Relu_ts[5,12]
   - A[3,6] * Conv_0_Relu_ts[6,12] - A[3,7] * Conv_0_Relu_ts[7,12]
   + A[4,3] * Conv_1_aggregated_features[3,12]
   + A[5,3] * Conv_1_aggregated_features[3,12]
   + A[6,3] * Conv_1_aggregated_features[3,12]
   + A[7,3] * Conv_1_aggregated_features[3,12] ] = 0
 Conv_1_averages_constraint[3,13]: [
   A[0,3] * Conv_1_aggregated_features[3,13]
   + A[1,3] * Conv_1_aggregated_features[3,13]
   + A[2,3] * Conv_1_aggregated_features[3,13]
   - A[3,0] * Conv_0_Relu_ts[0,13] - A[3,1] * Conv_0_Relu_ts[1,13]
   - A[3,2] * Conv_0_Relu_ts[2,13] - A[3,3] * Conv_0_Relu_ts[3,13]
   + A[3,3] * Conv_1_aggregated_features[3,13]
   - A[3,4] * Conv_0_Relu_ts[4,13] - A[3,5] * Conv_0_Relu_ts[5,13]
   - A[3,6] * Conv_0_Relu_ts[6,13] - A[3,7] * Conv_0_Relu_ts[7,13]
   + A[4,3] * Conv_1_aggregated_features[3,13]
   + A[5,3] * Conv_1_aggregated_features[3,13]
   + A[6,3] * Conv_1_aggregated_features[3,13]
   + A[7,3] * Conv_1_aggregated_features[3,13] ] = 0
 Conv_1_averages_constraint[3,14]: [
   A[0,3] * Conv_1_aggregated_features[3,14]
   + A[1,3] * Conv_1_aggregated_features[3,14]
   + A[2,3] * Conv_1_aggregated_features[3,14]
   - A[3,0] * Conv_0_Relu_ts[0,14] - A[3,1] * Conv_0_Relu_ts[1,14]
   - A[3,2] * Conv_0_Relu_ts[2,14] - A[3,3] * Conv_0_Relu_ts[3,14]
   + A[3,3] * Conv_1_aggregated_features[3,14]
   - A[3,4] * Conv_0_Relu_ts[4,14] - A[3,5] * Conv_0_Relu_ts[5,14]
   - A[3,6] * Conv_0_Relu_ts[6,14] - A[3,7] * Conv_0_Relu_ts[7,14]
   + A[4,3] * Conv_1_aggregated_features[3,14]
   + A[5,3] * Conv_1_aggregated_features[3,14]
   + A[6,3] * Conv_1_aggregated_features[3,14]
   + A[7,3] * Conv_1_aggregated_features[3,14] ] = 0
 Conv_1_averages_constraint[3,15]: [
   A[0,3] * Conv_1_aggregated_features[3,15]
   + A[1,3] * Conv_1_aggregated_features[3,15]
   + A[2,3] * Conv_1_aggregated_features[3,15]
   - A[3,0] * Conv_0_Relu_ts[0,15] - A[3,1] * Conv_0_Relu_ts[1,15]
   - A[3,2] * Conv_0_Relu_ts[2,15] - A[3,3] * Conv_0_Relu_ts[3,15]
   + A[3,3] * Conv_1_aggregated_features[3,15]
   - A[3,4] * Conv_0_Relu_ts[4,15] - A[3,5] * Conv_0_Relu_ts[5,15]
   - A[3,6] * Conv_0_Relu_ts[6,15] - A[3,7] * Conv_0_Relu_ts[7,15]
   + A[4,3] * Conv_1_aggregated_features[3,15]
   + A[5,3] * Conv_1_aggregated_features[3,15]
   + A[6,3] * Conv_1_aggregated_features[3,15]
   + A[7,3] * Conv_1_aggregated_features[3,15] ] = 0
 Conv_1_averages_constraint[4,0]: [
   A[0,4] * Conv_1_aggregated_features[4,0]
   + A[1,4] * Conv_1_aggregated_features[4,0]
   + A[2,4] * Conv_1_aggregated_features[4,0]
   + A[3,4] * Conv_1_aggregated_features[4,0]
   - A[4,0] * Conv_0_Relu_ts[0,0] - A[4,1] * Conv_0_Relu_ts[1,0]
   - A[4,2] * Conv_0_Relu_ts[2,0] - A[4,3] * Conv_0_Relu_ts[3,0]
   - A[4,4] * Conv_0_Relu_ts[4,0]
   + A[4,4] * Conv_1_aggregated_features[4,0]
   - A[4,5] * Conv_0_Relu_ts[5,0] - A[4,6] * Conv_0_Relu_ts[6,0]
   - A[4,7] * Conv_0_Relu_ts[7,0]
   + A[5,4] * Conv_1_aggregated_features[4,0]
   + A[6,4] * Conv_1_aggregated_features[4,0]
   + A[7,4] * Conv_1_aggregated_features[4,0] ] = 0
 Conv_1_averages_constraint[4,1]: [
   A[0,4] * Conv_1_aggregated_features[4,1]
   + A[1,4] * Conv_1_aggregated_features[4,1]
   + A[2,4] * Conv_1_aggregated_features[4,1]
   + A[3,4] * Conv_1_aggregated_features[4,1]
   - A[4,0] * Conv_0_Relu_ts[0,1] - A[4,1] * Conv_0_Relu_ts[1,1]
   - A[4,2] * Conv_0_Relu_ts[2,1] - A[4,3] * Conv_0_Relu_ts[3,1]
   - A[4,4] * Conv_0_Relu_ts[4,1]
   + A[4,4] * Conv_1_aggregated_features[4,1]
   - A[4,5] * Conv_0_Relu_ts[5,1] - A[4,6] * Conv_0_Relu_ts[6,1]
   - A[4,7] * Conv_0_Relu_ts[7,1]
   + A[5,4] * Conv_1_aggregated_features[4,1]
   + A[6,4] * Conv_1_aggregated_features[4,1]
   + A[7,4] * Conv_1_aggregated_features[4,1] ] = 0
 Conv_1_averages_constraint[4,2]: [
   A[0,4] * Conv_1_aggregated_features[4,2]
   + A[1,4] * Conv_1_aggregated_features[4,2]
   + A[2,4] * Conv_1_aggregated_features[4,2]
   + A[3,4] * Conv_1_aggregated_features[4,2]
   - A[4,0] * Conv_0_Relu_ts[0,2] - A[4,1] * Conv_0_Relu_ts[1,2]
   - A[4,2] * Conv_0_Relu_ts[2,2] - A[4,3] * Conv_0_Relu_ts[3,2]
   - A[4,4] * Conv_0_Relu_ts[4,2]
   + A[4,4] * Conv_1_aggregated_features[4,2]
   - A[4,5] * Conv_0_Relu_ts[5,2] - A[4,6] * Conv_0_Relu_ts[6,2]
   - A[4,7] * Conv_0_Relu_ts[7,2]
   + A[5,4] * Conv_1_aggregated_features[4,2]
   + A[6,4] * Conv_1_aggregated_features[4,2]
   + A[7,4] * Conv_1_aggregated_features[4,2] ] = 0
 Conv_1_averages_constraint[4,3]: [
   A[0,4] * Conv_1_aggregated_features[4,3]
   + A[1,4] * Conv_1_aggregated_features[4,3]
   + A[2,4] * Conv_1_aggregated_features[4,3]
   + A[3,4] * Conv_1_aggregated_features[4,3]
   - A[4,0] * Conv_0_Relu_ts[0,3] - A[4,1] * Conv_0_Relu_ts[1,3]
   - A[4,2] * Conv_0_Relu_ts[2,3] - A[4,3] * Conv_0_Relu_ts[3,3]
   - A[4,4] * Conv_0_Relu_ts[4,3]
   + A[4,4] * Conv_1_aggregated_features[4,3]
   - A[4,5] * Conv_0_Relu_ts[5,3] - A[4,6] * Conv_0_Relu_ts[6,3]
   - A[4,7] * Conv_0_Relu_ts[7,3]
   + A[5,4] * Conv_1_aggregated_features[4,3]
   + A[6,4] * Conv_1_aggregated_features[4,3]
   + A[7,4] * Conv_1_aggregated_features[4,3] ] = 0
 Conv_1_averages_constraint[4,4]: [
   A[0,4] * Conv_1_aggregated_features[4,4]
   + A[1,4] * Conv_1_aggregated_features[4,4]
   + A[2,4] * Conv_1_aggregated_features[4,4]
   + A[3,4] * Conv_1_aggregated_features[4,4]
   - A[4,0] * Conv_0_Relu_ts[0,4] - A[4,1] * Conv_0_Relu_ts[1,4]
   - A[4,2] * Conv_0_Relu_ts[2,4] - A[4,3] * Conv_0_Relu_ts[3,4]
   - A[4,4] * Conv_0_Relu_ts[4,4]
   + A[4,4] * Conv_1_aggregated_features[4,4]
   - A[4,5] * Conv_0_Relu_ts[5,4] - A[4,6] * Conv_0_Relu_ts[6,4]
   - A[4,7] * Conv_0_Relu_ts[7,4]
   + A[5,4] * Conv_1_aggregated_features[4,4]
   + A[6,4] * Conv_1_aggregated_features[4,4]
   + A[7,4] * Conv_1_aggregated_features[4,4] ] = 0
 Conv_1_averages_constraint[4,5]: [
   A[0,4] * Conv_1_aggregated_features[4,5]
   + A[1,4] * Conv_1_aggregated_features[4,5]
   + A[2,4] * Conv_1_aggregated_features[4,5]
   + A[3,4] * Conv_1_aggregated_features[4,5]
   - A[4,0] * Conv_0_Relu_ts[0,5] - A[4,1] * Conv_0_Relu_ts[1,5]
   - A[4,2] * Conv_0_Relu_ts[2,5] - A[4,3] * Conv_0_Relu_ts[3,5]
   - A[4,4] * Conv_0_Relu_ts[4,5]
   + A[4,4] * Conv_1_aggregated_features[4,5]
   - A[4,5] * Conv_0_Relu_ts[5,5] - A[4,6] * Conv_0_Relu_ts[6,5]
   - A[4,7] * Conv_0_Relu_ts[7,5]
   + A[5,4] * Conv_1_aggregated_features[4,5]
   + A[6,4] * Conv_1_aggregated_features[4,5]
   + A[7,4] * Conv_1_aggregated_features[4,5] ] = 0
 Conv_1_averages_constraint[4,6]: [
   A[0,4] * Conv_1_aggregated_features[4,6]
   + A[1,4] * Conv_1_aggregated_features[4,6]
   + A[2,4] * Conv_1_aggregated_features[4,6]
   + A[3,4] * Conv_1_aggregated_features[4,6]
   - A[4,0] * Conv_0_Relu_ts[0,6] - A[4,1] * Conv_0_Relu_ts[1,6]
   - A[4,2] * Conv_0_Relu_ts[2,6] - A[4,3] * Conv_0_Relu_ts[3,6]
   - A[4,4] * Conv_0_Relu_ts[4,6]
   + A[4,4] * Conv_1_aggregated_features[4,6]
   - A[4,5] * Conv_0_Relu_ts[5,6] - A[4,6] * Conv_0_Relu_ts[6,6]
   - A[4,7] * Conv_0_Relu_ts[7,6]
   + A[5,4] * Conv_1_aggregated_features[4,6]
   + A[6,4] * Conv_1_aggregated_features[4,6]
   + A[7,4] * Conv_1_aggregated_features[4,6] ] = 0
 Conv_1_averages_constraint[4,7]: [
   A[0,4] * Conv_1_aggregated_features[4,7]
   + A[1,4] * Conv_1_aggregated_features[4,7]
   + A[2,4] * Conv_1_aggregated_features[4,7]
   + A[3,4] * Conv_1_aggregated_features[4,7]
   - A[4,0] * Conv_0_Relu_ts[0,7] - A[4,1] * Conv_0_Relu_ts[1,7]
   - A[4,2] * Conv_0_Relu_ts[2,7] - A[4,3] * Conv_0_Relu_ts[3,7]
   - A[4,4] * Conv_0_Relu_ts[4,7]
   + A[4,4] * Conv_1_aggregated_features[4,7]
   - A[4,5] * Conv_0_Relu_ts[5,7] - A[4,6] * Conv_0_Relu_ts[6,7]
   - A[4,7] * Conv_0_Relu_ts[7,7]
   + A[5,4] * Conv_1_aggregated_features[4,7]
   + A[6,4] * Conv_1_aggregated_features[4,7]
   + A[7,4] * Conv_1_aggregated_features[4,7] ] = 0
 Conv_1_averages_constraint[4,8]: [
   A[0,4] * Conv_1_aggregated_features[4,8]
   + A[1,4] * Conv_1_aggregated_features[4,8]
   + A[2,4] * Conv_1_aggregated_features[4,8]
   + A[3,4] * Conv_1_aggregated_features[4,8]
   - A[4,0] * Conv_0_Relu_ts[0,8] - A[4,1] * Conv_0_Relu_ts[1,8]
   - A[4,2] * Conv_0_Relu_ts[2,8] - A[4,3] * Conv_0_Relu_ts[3,8]
   - A[4,4] * Conv_0_Relu_ts[4,8]
   + A[4,4] * Conv_1_aggregated_features[4,8]
   - A[4,5] * Conv_0_Relu_ts[5,8] - A[4,6] * Conv_0_Relu_ts[6,8]
   - A[4,7] * Conv_0_Relu_ts[7,8]
   + A[5,4] * Conv_1_aggregated_features[4,8]
   + A[6,4] * Conv_1_aggregated_features[4,8]
   + A[7,4] * Conv_1_aggregated_features[4,8] ] = 0
 Conv_1_averages_constraint[4,9]: [
   A[0,4] * Conv_1_aggregated_features[4,9]
   + A[1,4] * Conv_1_aggregated_features[4,9]
   + A[2,4] * Conv_1_aggregated_features[4,9]
   + A[3,4] * Conv_1_aggregated_features[4,9]
   - A[4,0] * Conv_0_Relu_ts[0,9] - A[4,1] * Conv_0_Relu_ts[1,9]
   - A[4,2] * Conv_0_Relu_ts[2,9] - A[4,3] * Conv_0_Relu_ts[3,9]
   - A[4,4] * Conv_0_Relu_ts[4,9]
   + A[4,4] * Conv_1_aggregated_features[4,9]
   - A[4,5] * Conv_0_Relu_ts[5,9] - A[4,6] * Conv_0_Relu_ts[6,9]
   - A[4,7] * Conv_0_Relu_ts[7,9]
   + A[5,4] * Conv_1_aggregated_features[4,9]
   + A[6,4] * Conv_1_aggregated_features[4,9]
   + A[7,4] * Conv_1_aggregated_features[4,9] ] = 0
 Conv_1_averages_constraint[4,10]: [
   A[0,4] * Conv_1_aggregated_features[4,10]
   + A[1,4] * Conv_1_aggregated_features[4,10]
   + A[2,4] * Conv_1_aggregated_features[4,10]
   + A[3,4] * Conv_1_aggregated_features[4,10]
   - A[4,0] * Conv_0_Relu_ts[0,10] - A[4,1] * Conv_0_Relu_ts[1,10]
   - A[4,2] * Conv_0_Relu_ts[2,10] - A[4,3] * Conv_0_Relu_ts[3,10]
   - A[4,4] * Conv_0_Relu_ts[4,10]
   + A[4,4] * Conv_1_aggregated_features[4,10]
   - A[4,5] * Conv_0_Relu_ts[5,10] - A[4,6] * Conv_0_Relu_ts[6,10]
   - A[4,7] * Conv_0_Relu_ts[7,10]
   + A[5,4] * Conv_1_aggregated_features[4,10]
   + A[6,4] * Conv_1_aggregated_features[4,10]
   + A[7,4] * Conv_1_aggregated_features[4,10] ] = 0
 Conv_1_averages_constraint[4,11]: [
   A[0,4] * Conv_1_aggregated_features[4,11]
   + A[1,4] * Conv_1_aggregated_features[4,11]
   + A[2,4] * Conv_1_aggregated_features[4,11]
   + A[3,4] * Conv_1_aggregated_features[4,11]
   - A[4,0] * Conv_0_Relu_ts[0,11] - A[4,1] * Conv_0_Relu_ts[1,11]
   - A[4,2] * Conv_0_Relu_ts[2,11] - A[4,3] * Conv_0_Relu_ts[3,11]
   - A[4,4] * Conv_0_Relu_ts[4,11]
   + A[4,4] * Conv_1_aggregated_features[4,11]
   - A[4,5] * Conv_0_Relu_ts[5,11] - A[4,6] * Conv_0_Relu_ts[6,11]
   - A[4,7] * Conv_0_Relu_ts[7,11]
   + A[5,4] * Conv_1_aggregated_features[4,11]
   + A[6,4] * Conv_1_aggregated_features[4,11]
   + A[7,4] * Conv_1_aggregated_features[4,11] ] = 0
 Conv_1_averages_constraint[4,12]: [
   A[0,4] * Conv_1_aggregated_features[4,12]
   + A[1,4] * Conv_1_aggregated_features[4,12]
   + A[2,4] * Conv_1_aggregated_features[4,12]
   + A[3,4] * Conv_1_aggregated_features[4,12]
   - A[4,0] * Conv_0_Relu_ts[0,12] - A[4,1] * Conv_0_Relu_ts[1,12]
   - A[4,2] * Conv_0_Relu_ts[2,12] - A[4,3] * Conv_0_Relu_ts[3,12]
   - A[4,4] * Conv_0_Relu_ts[4,12]
   + A[4,4] * Conv_1_aggregated_features[4,12]
   - A[4,5] * Conv_0_Relu_ts[5,12] - A[4,6] * Conv_0_Relu_ts[6,12]
   - A[4,7] * Conv_0_Relu_ts[7,12]
   + A[5,4] * Conv_1_aggregated_features[4,12]
   + A[6,4] * Conv_1_aggregated_features[4,12]
   + A[7,4] * Conv_1_aggregated_features[4,12] ] = 0
 Conv_1_averages_constraint[4,13]: [
   A[0,4] * Conv_1_aggregated_features[4,13]
   + A[1,4] * Conv_1_aggregated_features[4,13]
   + A[2,4] * Conv_1_aggregated_features[4,13]
   + A[3,4] * Conv_1_aggregated_features[4,13]
   - A[4,0] * Conv_0_Relu_ts[0,13] - A[4,1] * Conv_0_Relu_ts[1,13]
   - A[4,2] * Conv_0_Relu_ts[2,13] - A[4,3] * Conv_0_Relu_ts[3,13]
   - A[4,4] * Conv_0_Relu_ts[4,13]
   + A[4,4] * Conv_1_aggregated_features[4,13]
   - A[4,5] * Conv_0_Relu_ts[5,13] - A[4,6] * Conv_0_Relu_ts[6,13]
   - A[4,7] * Conv_0_Relu_ts[7,13]
   + A[5,4] * Conv_1_aggregated_features[4,13]
   + A[6,4] * Conv_1_aggregated_features[4,13]
   + A[7,4] * Conv_1_aggregated_features[4,13] ] = 0
 Conv_1_averages_constraint[4,14]: [
   A[0,4] * Conv_1_aggregated_features[4,14]
   + A[1,4] * Conv_1_aggregated_features[4,14]
   + A[2,4] * Conv_1_aggregated_features[4,14]
   + A[3,4] * Conv_1_aggregated_features[4,14]
   - A[4,0] * Conv_0_Relu_ts[0,14] - A[4,1] * Conv_0_Relu_ts[1,14]
   - A[4,2] * Conv_0_Relu_ts[2,14] - A[4,3] * Conv_0_Relu_ts[3,14]
   - A[4,4] * Conv_0_Relu_ts[4,14]
   + A[4,4] * Conv_1_aggregated_features[4,14]
   - A[4,5] * Conv_0_Relu_ts[5,14] - A[4,6] * Conv_0_Relu_ts[6,14]
   - A[4,7] * Conv_0_Relu_ts[7,14]
   + A[5,4] * Conv_1_aggregated_features[4,14]
   + A[6,4] * Conv_1_aggregated_features[4,14]
   + A[7,4] * Conv_1_aggregated_features[4,14] ] = 0
 Conv_1_averages_constraint[4,15]: [
   A[0,4] * Conv_1_aggregated_features[4,15]
   + A[1,4] * Conv_1_aggregated_features[4,15]
   + A[2,4] * Conv_1_aggregated_features[4,15]
   + A[3,4] * Conv_1_aggregated_features[4,15]
   - A[4,0] * Conv_0_Relu_ts[0,15] - A[4,1] * Conv_0_Relu_ts[1,15]
   - A[4,2] * Conv_0_Relu_ts[2,15] - A[4,3] * Conv_0_Relu_ts[3,15]
   - A[4,4] * Conv_0_Relu_ts[4,15]
   + A[4,4] * Conv_1_aggregated_features[4,15]
   - A[4,5] * Conv_0_Relu_ts[5,15] - A[4,6] * Conv_0_Relu_ts[6,15]
   - A[4,7] * Conv_0_Relu_ts[7,15]
   + A[5,4] * Conv_1_aggregated_features[4,15]
   + A[6,4] * Conv_1_aggregated_features[4,15]
   + A[7,4] * Conv_1_aggregated_features[4,15] ] = 0
 Conv_1_averages_constraint[5,0]: [
   A[0,5] * Conv_1_aggregated_features[5,0]
   + A[1,5] * Conv_1_aggregated_features[5,0]
   + A[2,5] * Conv_1_aggregated_features[5,0]
   + A[3,5] * Conv_1_aggregated_features[5,0]
   + A[4,5] * Conv_1_aggregated_features[5,0]
   - A[5,0] * Conv_0_Relu_ts[0,0] - A[5,1] * Conv_0_Relu_ts[1,0]
   - A[5,2] * Conv_0_Relu_ts[2,0] - A[5,3] * Conv_0_Relu_ts[3,0]
   - A[5,4] * Conv_0_Relu_ts[4,0] - A[5,5] * Conv_0_Relu_ts[5,0]
   + A[5,5] * Conv_1_aggregated_features[5,0]
   - A[5,6] * Conv_0_Relu_ts[6,0] - A[5,7] * Conv_0_Relu_ts[7,0]
   + A[6,5] * Conv_1_aggregated_features[5,0]
   + A[7,5] * Conv_1_aggregated_features[5,0] ] = 0
 Conv_1_averages_constraint[5,1]: [
   A[0,5] * Conv_1_aggregated_features[5,1]
   + A[1,5] * Conv_1_aggregated_features[5,1]
   + A[2,5] * Conv_1_aggregated_features[5,1]
   + A[3,5] * Conv_1_aggregated_features[5,1]
   + A[4,5] * Conv_1_aggregated_features[5,1]
   - A[5,0] * Conv_0_Relu_ts[0,1] - A[5,1] * Conv_0_Relu_ts[1,1]
   - A[5,2] * Conv_0_Relu_ts[2,1] - A[5,3] * Conv_0_Relu_ts[3,1]
   - A[5,4] * Conv_0_Relu_ts[4,1] - A[5,5] * Conv_0_Relu_ts[5,1]
   + A[5,5] * Conv_1_aggregated_features[5,1]
   - A[5,6] * Conv_0_Relu_ts[6,1] - A[5,7] * Conv_0_Relu_ts[7,1]
   + A[6,5] * Conv_1_aggregated_features[5,1]
   + A[7,5] * Conv_1_aggregated_features[5,1] ] = 0
 Conv_1_averages_constraint[5,2]: [
   A[0,5] * Conv_1_aggregated_features[5,2]
   + A[1,5] * Conv_1_aggregated_features[5,2]
   + A[2,5] * Conv_1_aggregated_features[5,2]
   + A[3,5] * Conv_1_aggregated_features[5,2]
   + A[4,5] * Conv_1_aggregated_features[5,2]
   - A[5,0] * Conv_0_Relu_ts[0,2] - A[5,1] * Conv_0_Relu_ts[1,2]
   - A[5,2] * Conv_0_Relu_ts[2,2] - A[5,3] * Conv_0_Relu_ts[3,2]
   - A[5,4] * Conv_0_Relu_ts[4,2] - A[5,5] * Conv_0_Relu_ts[5,2]
   + A[5,5] * Conv_1_aggregated_features[5,2]
   - A[5,6] * Conv_0_Relu_ts[6,2] - A[5,7] * Conv_0_Relu_ts[7,2]
   + A[6,5] * Conv_1_aggregated_features[5,2]
   + A[7,5] * Conv_1_aggregated_features[5,2] ] = 0
 Conv_1_averages_constraint[5,3]: [
   A[0,5] * Conv_1_aggregated_features[5,3]
   + A[1,5] * Conv_1_aggregated_features[5,3]
   + A[2,5] * Conv_1_aggregated_features[5,3]
   + A[3,5] * Conv_1_aggregated_features[5,3]
   + A[4,5] * Conv_1_aggregated_features[5,3]
   - A[5,0] * Conv_0_Relu_ts[0,3] - A[5,1] * Conv_0_Relu_ts[1,3]
   - A[5,2] * Conv_0_Relu_ts[2,3] - A[5,3] * Conv_0_Relu_ts[3,3]
   - A[5,4] * Conv_0_Relu_ts[4,3] - A[5,5] * Conv_0_Relu_ts[5,3]
   + A[5,5] * Conv_1_aggregated_features[5,3]
   - A[5,6] * Conv_0_Relu_ts[6,3] - A[5,7] * Conv_0_Relu_ts[7,3]
   + A[6,5] * Conv_1_aggregated_features[5,3]
   + A[7,5] * Conv_1_aggregated_features[5,3] ] = 0
 Conv_1_averages_constraint[5,4]: [
   A[0,5] * Conv_1_aggregated_features[5,4]
   + A[1,5] * Conv_1_aggregated_features[5,4]
   + A[2,5] * Conv_1_aggregated_features[5,4]
   + A[3,5] * Conv_1_aggregated_features[5,4]
   + A[4,5] * Conv_1_aggregated_features[5,4]
   - A[5,0] * Conv_0_Relu_ts[0,4] - A[5,1] * Conv_0_Relu_ts[1,4]
   - A[5,2] * Conv_0_Relu_ts[2,4] - A[5,3] * Conv_0_Relu_ts[3,4]
   - A[5,4] * Conv_0_Relu_ts[4,4] - A[5,5] * Conv_0_Relu_ts[5,4]
   + A[5,5] * Conv_1_aggregated_features[5,4]
   - A[5,6] * Conv_0_Relu_ts[6,4] - A[5,7] * Conv_0_Relu_ts[7,4]
   + A[6,5] * Conv_1_aggregated_features[5,4]
   + A[7,5] * Conv_1_aggregated_features[5,4] ] = 0
 Conv_1_averages_constraint[5,5]: [
   A[0,5] * Conv_1_aggregated_features[5,5]
   + A[1,5] * Conv_1_aggregated_features[5,5]
   + A[2,5] * Conv_1_aggregated_features[5,5]
   + A[3,5] * Conv_1_aggregated_features[5,5]
   + A[4,5] * Conv_1_aggregated_features[5,5]
   - A[5,0] * Conv_0_Relu_ts[0,5] - A[5,1] * Conv_0_Relu_ts[1,5]
   - A[5,2] * Conv_0_Relu_ts[2,5] - A[5,3] * Conv_0_Relu_ts[3,5]
   - A[5,4] * Conv_0_Relu_ts[4,5] - A[5,5] * Conv_0_Relu_ts[5,5]
   + A[5,5] * Conv_1_aggregated_features[5,5]
   - A[5,6] * Conv_0_Relu_ts[6,5] - A[5,7] * Conv_0_Relu_ts[7,5]
   + A[6,5] * Conv_1_aggregated_features[5,5]
   + A[7,5] * Conv_1_aggregated_features[5,5] ] = 0
 Conv_1_averages_constraint[5,6]: [
   A[0,5] * Conv_1_aggregated_features[5,6]
   + A[1,5] * Conv_1_aggregated_features[5,6]
   + A[2,5] * Conv_1_aggregated_features[5,6]
   + A[3,5] * Conv_1_aggregated_features[5,6]
   + A[4,5] * Conv_1_aggregated_features[5,6]
   - A[5,0] * Conv_0_Relu_ts[0,6] - A[5,1] * Conv_0_Relu_ts[1,6]
   - A[5,2] * Conv_0_Relu_ts[2,6] - A[5,3] * Conv_0_Relu_ts[3,6]
   - A[5,4] * Conv_0_Relu_ts[4,6] - A[5,5] * Conv_0_Relu_ts[5,6]
   + A[5,5] * Conv_1_aggregated_features[5,6]
   - A[5,6] * Conv_0_Relu_ts[6,6] - A[5,7] * Conv_0_Relu_ts[7,6]
   + A[6,5] * Conv_1_aggregated_features[5,6]
   + A[7,5] * Conv_1_aggregated_features[5,6] ] = 0
 Conv_1_averages_constraint[5,7]: [
   A[0,5] * Conv_1_aggregated_features[5,7]
   + A[1,5] * Conv_1_aggregated_features[5,7]
   + A[2,5] * Conv_1_aggregated_features[5,7]
   + A[3,5] * Conv_1_aggregated_features[5,7]
   + A[4,5] * Conv_1_aggregated_features[5,7]
   - A[5,0] * Conv_0_Relu_ts[0,7] - A[5,1] * Conv_0_Relu_ts[1,7]
   - A[5,2] * Conv_0_Relu_ts[2,7] - A[5,3] * Conv_0_Relu_ts[3,7]
   - A[5,4] * Conv_0_Relu_ts[4,7] - A[5,5] * Conv_0_Relu_ts[5,7]
   + A[5,5] * Conv_1_aggregated_features[5,7]
   - A[5,6] * Conv_0_Relu_ts[6,7] - A[5,7] * Conv_0_Relu_ts[7,7]
   + A[6,5] * Conv_1_aggregated_features[5,7]
   + A[7,5] * Conv_1_aggregated_features[5,7] ] = 0
 Conv_1_averages_constraint[5,8]: [
   A[0,5] * Conv_1_aggregated_features[5,8]
   + A[1,5] * Conv_1_aggregated_features[5,8]
   + A[2,5] * Conv_1_aggregated_features[5,8]
   + A[3,5] * Conv_1_aggregated_features[5,8]
   + A[4,5] * Conv_1_aggregated_features[5,8]
   - A[5,0] * Conv_0_Relu_ts[0,8] - A[5,1] * Conv_0_Relu_ts[1,8]
   - A[5,2] * Conv_0_Relu_ts[2,8] - A[5,3] * Conv_0_Relu_ts[3,8]
   - A[5,4] * Conv_0_Relu_ts[4,8] - A[5,5] * Conv_0_Relu_ts[5,8]
   + A[5,5] * Conv_1_aggregated_features[5,8]
   - A[5,6] * Conv_0_Relu_ts[6,8] - A[5,7] * Conv_0_Relu_ts[7,8]
   + A[6,5] * Conv_1_aggregated_features[5,8]
   + A[7,5] * Conv_1_aggregated_features[5,8] ] = 0
 Conv_1_averages_constraint[5,9]: [
   A[0,5] * Conv_1_aggregated_features[5,9]
   + A[1,5] * Conv_1_aggregated_features[5,9]
   + A[2,5] * Conv_1_aggregated_features[5,9]
   + A[3,5] * Conv_1_aggregated_features[5,9]
   + A[4,5] * Conv_1_aggregated_features[5,9]
   - A[5,0] * Conv_0_Relu_ts[0,9] - A[5,1] * Conv_0_Relu_ts[1,9]
   - A[5,2] * Conv_0_Relu_ts[2,9] - A[5,3] * Conv_0_Relu_ts[3,9]
   - A[5,4] * Conv_0_Relu_ts[4,9] - A[5,5] * Conv_0_Relu_ts[5,9]
   + A[5,5] * Conv_1_aggregated_features[5,9]
   - A[5,6] * Conv_0_Relu_ts[6,9] - A[5,7] * Conv_0_Relu_ts[7,9]
   + A[6,5] * Conv_1_aggregated_features[5,9]
   + A[7,5] * Conv_1_aggregated_features[5,9] ] = 0
 Conv_1_averages_constraint[5,10]: [
   A[0,5] * Conv_1_aggregated_features[5,10]
   + A[1,5] * Conv_1_aggregated_features[5,10]
   + A[2,5] * Conv_1_aggregated_features[5,10]
   + A[3,5] * Conv_1_aggregated_features[5,10]
   + A[4,5] * Conv_1_aggregated_features[5,10]
   - A[5,0] * Conv_0_Relu_ts[0,10] - A[5,1] * Conv_0_Relu_ts[1,10]
   - A[5,2] * Conv_0_Relu_ts[2,10] - A[5,3] * Conv_0_Relu_ts[3,10]
   - A[5,4] * Conv_0_Relu_ts[4,10] - A[5,5] * Conv_0_Relu_ts[5,10]
   + A[5,5] * Conv_1_aggregated_features[5,10]
   - A[5,6] * Conv_0_Relu_ts[6,10] - A[5,7] * Conv_0_Relu_ts[7,10]
   + A[6,5] * Conv_1_aggregated_features[5,10]
   + A[7,5] * Conv_1_aggregated_features[5,10] ] = 0
 Conv_1_averages_constraint[5,11]: [
   A[0,5] * Conv_1_aggregated_features[5,11]
   + A[1,5] * Conv_1_aggregated_features[5,11]
   + A[2,5] * Conv_1_aggregated_features[5,11]
   + A[3,5] * Conv_1_aggregated_features[5,11]
   + A[4,5] * Conv_1_aggregated_features[5,11]
   - A[5,0] * Conv_0_Relu_ts[0,11] - A[5,1] * Conv_0_Relu_ts[1,11]
   - A[5,2] * Conv_0_Relu_ts[2,11] - A[5,3] * Conv_0_Relu_ts[3,11]
   - A[5,4] * Conv_0_Relu_ts[4,11] - A[5,5] * Conv_0_Relu_ts[5,11]
   + A[5,5] * Conv_1_aggregated_features[5,11]
   - A[5,6] * Conv_0_Relu_ts[6,11] - A[5,7] * Conv_0_Relu_ts[7,11]
   + A[6,5] * Conv_1_aggregated_features[5,11]
   + A[7,5] * Conv_1_aggregated_features[5,11] ] = 0
 Conv_1_averages_constraint[5,12]: [
   A[0,5] * Conv_1_aggregated_features[5,12]
   + A[1,5] * Conv_1_aggregated_features[5,12]
   + A[2,5] * Conv_1_aggregated_features[5,12]
   + A[3,5] * Conv_1_aggregated_features[5,12]
   + A[4,5] * Conv_1_aggregated_features[5,12]
   - A[5,0] * Conv_0_Relu_ts[0,12] - A[5,1] * Conv_0_Relu_ts[1,12]
   - A[5,2] * Conv_0_Relu_ts[2,12] - A[5,3] * Conv_0_Relu_ts[3,12]
   - A[5,4] * Conv_0_Relu_ts[4,12] - A[5,5] * Conv_0_Relu_ts[5,12]
   + A[5,5] * Conv_1_aggregated_features[5,12]
   - A[5,6] * Conv_0_Relu_ts[6,12] - A[5,7] * Conv_0_Relu_ts[7,12]
   + A[6,5] * Conv_1_aggregated_features[5,12]
   + A[7,5] * Conv_1_aggregated_features[5,12] ] = 0
 Conv_1_averages_constraint[5,13]: [
   A[0,5] * Conv_1_aggregated_features[5,13]
   + A[1,5] * Conv_1_aggregated_features[5,13]
   + A[2,5] * Conv_1_aggregated_features[5,13]
   + A[3,5] * Conv_1_aggregated_features[5,13]
   + A[4,5] * Conv_1_aggregated_features[5,13]
   - A[5,0] * Conv_0_Relu_ts[0,13] - A[5,1] * Conv_0_Relu_ts[1,13]
   - A[5,2] * Conv_0_Relu_ts[2,13] - A[5,3] * Conv_0_Relu_ts[3,13]
   - A[5,4] * Conv_0_Relu_ts[4,13] - A[5,5] * Conv_0_Relu_ts[5,13]
   + A[5,5] * Conv_1_aggregated_features[5,13]
   - A[5,6] * Conv_0_Relu_ts[6,13] - A[5,7] * Conv_0_Relu_ts[7,13]
   + A[6,5] * Conv_1_aggregated_features[5,13]
   + A[7,5] * Conv_1_aggregated_features[5,13] ] = 0
 Conv_1_averages_constraint[5,14]: [
   A[0,5] * Conv_1_aggregated_features[5,14]
   + A[1,5] * Conv_1_aggregated_features[5,14]
   + A[2,5] * Conv_1_aggregated_features[5,14]
   + A[3,5] * Conv_1_aggregated_features[5,14]
   + A[4,5] * Conv_1_aggregated_features[5,14]
   - A[5,0] * Conv_0_Relu_ts[0,14] - A[5,1] * Conv_0_Relu_ts[1,14]
   - A[5,2] * Conv_0_Relu_ts[2,14] - A[5,3] * Conv_0_Relu_ts[3,14]
   - A[5,4] * Conv_0_Relu_ts[4,14] - A[5,5] * Conv_0_Relu_ts[5,14]
   + A[5,5] * Conv_1_aggregated_features[5,14]
   - A[5,6] * Conv_0_Relu_ts[6,14] - A[5,7] * Conv_0_Relu_ts[7,14]
   + A[6,5] * Conv_1_aggregated_features[5,14]
   + A[7,5] * Conv_1_aggregated_features[5,14] ] = 0
 Conv_1_averages_constraint[5,15]: [
   A[0,5] * Conv_1_aggregated_features[5,15]
   + A[1,5] * Conv_1_aggregated_features[5,15]
   + A[2,5] * Conv_1_aggregated_features[5,15]
   + A[3,5] * Conv_1_aggregated_features[5,15]
   + A[4,5] * Conv_1_aggregated_features[5,15]
   - A[5,0] * Conv_0_Relu_ts[0,15] - A[5,1] * Conv_0_Relu_ts[1,15]
   - A[5,2] * Conv_0_Relu_ts[2,15] - A[5,3] * Conv_0_Relu_ts[3,15]
   - A[5,4] * Conv_0_Relu_ts[4,15] - A[5,5] * Conv_0_Relu_ts[5,15]
   + A[5,5] * Conv_1_aggregated_features[5,15]
   - A[5,6] * Conv_0_Relu_ts[6,15] - A[5,7] * Conv_0_Relu_ts[7,15]
   + A[6,5] * Conv_1_aggregated_features[5,15]
   + A[7,5] * Conv_1_aggregated_features[5,15] ] = 0
 Conv_1_averages_constraint[6,0]: [
   A[0,6] * Conv_1_aggregated_features[6,0]
   + A[1,6] * Conv_1_aggregated_features[6,0]
   + A[2,6] * Conv_1_aggregated_features[6,0]
   + A[3,6] * Conv_1_aggregated_features[6,0]
   + A[4,6] * Conv_1_aggregated_features[6,0]
   + A[5,6] * Conv_1_aggregated_features[6,0]
   - A[6,0] * Conv_0_Relu_ts[0,0] - A[6,1] * Conv_0_Relu_ts[1,0]
   - A[6,2] * Conv_0_Relu_ts[2,0] - A[6,3] * Conv_0_Relu_ts[3,0]
   - A[6,4] * Conv_0_Relu_ts[4,0] - A[6,5] * Conv_0_Relu_ts[5,0]
   - A[6,6] * Conv_0_Relu_ts[6,0]
   + A[6,6] * Conv_1_aggregated_features[6,0]
   - A[6,7] * Conv_0_Relu_ts[7,0]
   + A[7,6] * Conv_1_aggregated_features[6,0] ] = 0
 Conv_1_averages_constraint[6,1]: [
   A[0,6] * Conv_1_aggregated_features[6,1]
   + A[1,6] * Conv_1_aggregated_features[6,1]
   + A[2,6] * Conv_1_aggregated_features[6,1]
   + A[3,6] * Conv_1_aggregated_features[6,1]
   + A[4,6] * Conv_1_aggregated_features[6,1]
   + A[5,6] * Conv_1_aggregated_features[6,1]
   - A[6,0] * Conv_0_Relu_ts[0,1] - A[6,1] * Conv_0_Relu_ts[1,1]
   - A[6,2] * Conv_0_Relu_ts[2,1] - A[6,3] * Conv_0_Relu_ts[3,1]
   - A[6,4] * Conv_0_Relu_ts[4,1] - A[6,5] * Conv_0_Relu_ts[5,1]
   - A[6,6] * Conv_0_Relu_ts[6,1]
   + A[6,6] * Conv_1_aggregated_features[6,1]
   - A[6,7] * Conv_0_Relu_ts[7,1]
   + A[7,6] * Conv_1_aggregated_features[6,1] ] = 0
 Conv_1_averages_constraint[6,2]: [
   A[0,6] * Conv_1_aggregated_features[6,2]
   + A[1,6] * Conv_1_aggregated_features[6,2]
   + A[2,6] * Conv_1_aggregated_features[6,2]
   + A[3,6] * Conv_1_aggregated_features[6,2]
   + A[4,6] * Conv_1_aggregated_features[6,2]
   + A[5,6] * Conv_1_aggregated_features[6,2]
   - A[6,0] * Conv_0_Relu_ts[0,2] - A[6,1] * Conv_0_Relu_ts[1,2]
   - A[6,2] * Conv_0_Relu_ts[2,2] - A[6,3] * Conv_0_Relu_ts[3,2]
   - A[6,4] * Conv_0_Relu_ts[4,2] - A[6,5] * Conv_0_Relu_ts[5,2]
   - A[6,6] * Conv_0_Relu_ts[6,2]
   + A[6,6] * Conv_1_aggregated_features[6,2]
   - A[6,7] * Conv_0_Relu_ts[7,2]
   + A[7,6] * Conv_1_aggregated_features[6,2] ] = 0
 Conv_1_averages_constraint[6,3]: [
   A[0,6] * Conv_1_aggregated_features[6,3]
   + A[1,6] * Conv_1_aggregated_features[6,3]
   + A[2,6] * Conv_1_aggregated_features[6,3]
   + A[3,6] * Conv_1_aggregated_features[6,3]
   + A[4,6] * Conv_1_aggregated_features[6,3]
   + A[5,6] * Conv_1_aggregated_features[6,3]
   - A[6,0] * Conv_0_Relu_ts[0,3] - A[6,1] * Conv_0_Relu_ts[1,3]
   - A[6,2] * Conv_0_Relu_ts[2,3] - A[6,3] * Conv_0_Relu_ts[3,3]
   - A[6,4] * Conv_0_Relu_ts[4,3] - A[6,5] * Conv_0_Relu_ts[5,3]
   - A[6,6] * Conv_0_Relu_ts[6,3]
   + A[6,6] * Conv_1_aggregated_features[6,3]
   - A[6,7] * Conv_0_Relu_ts[7,3]
   + A[7,6] * Conv_1_aggregated_features[6,3] ] = 0
 Conv_1_averages_constraint[6,4]: [
   A[0,6] * Conv_1_aggregated_features[6,4]
   + A[1,6] * Conv_1_aggregated_features[6,4]
   + A[2,6] * Conv_1_aggregated_features[6,4]
   + A[3,6] * Conv_1_aggregated_features[6,4]
   + A[4,6] * Conv_1_aggregated_features[6,4]
   + A[5,6] * Conv_1_aggregated_features[6,4]
   - A[6,0] * Conv_0_Relu_ts[0,4] - A[6,1] * Conv_0_Relu_ts[1,4]
   - A[6,2] * Conv_0_Relu_ts[2,4] - A[6,3] * Conv_0_Relu_ts[3,4]
   - A[6,4] * Conv_0_Relu_ts[4,4] - A[6,5] * Conv_0_Relu_ts[5,4]
   - A[6,6] * Conv_0_Relu_ts[6,4]
   + A[6,6] * Conv_1_aggregated_features[6,4]
   - A[6,7] * Conv_0_Relu_ts[7,4]
   + A[7,6] * Conv_1_aggregated_features[6,4] ] = 0
 Conv_1_averages_constraint[6,5]: [
   A[0,6] * Conv_1_aggregated_features[6,5]
   + A[1,6] * Conv_1_aggregated_features[6,5]
   + A[2,6] * Conv_1_aggregated_features[6,5]
   + A[3,6] * Conv_1_aggregated_features[6,5]
   + A[4,6] * Conv_1_aggregated_features[6,5]
   + A[5,6] * Conv_1_aggregated_features[6,5]
   - A[6,0] * Conv_0_Relu_ts[0,5] - A[6,1] * Conv_0_Relu_ts[1,5]
   - A[6,2] * Conv_0_Relu_ts[2,5] - A[6,3] * Conv_0_Relu_ts[3,5]
   - A[6,4] * Conv_0_Relu_ts[4,5] - A[6,5] * Conv_0_Relu_ts[5,5]
   - A[6,6] * Conv_0_Relu_ts[6,5]
   + A[6,6] * Conv_1_aggregated_features[6,5]
   - A[6,7] * Conv_0_Relu_ts[7,5]
   + A[7,6] * Conv_1_aggregated_features[6,5] ] = 0
 Conv_1_averages_constraint[6,6]: [
   A[0,6] * Conv_1_aggregated_features[6,6]
   + A[1,6] * Conv_1_aggregated_features[6,6]
   + A[2,6] * Conv_1_aggregated_features[6,6]
   + A[3,6] * Conv_1_aggregated_features[6,6]
   + A[4,6] * Conv_1_aggregated_features[6,6]
   + A[5,6] * Conv_1_aggregated_features[6,6]
   - A[6,0] * Conv_0_Relu_ts[0,6] - A[6,1] * Conv_0_Relu_ts[1,6]
   - A[6,2] * Conv_0_Relu_ts[2,6] - A[6,3] * Conv_0_Relu_ts[3,6]
   - A[6,4] * Conv_0_Relu_ts[4,6] - A[6,5] * Conv_0_Relu_ts[5,6]
   - A[6,6] * Conv_0_Relu_ts[6,6]
   + A[6,6] * Conv_1_aggregated_features[6,6]
   - A[6,7] * Conv_0_Relu_ts[7,6]
   + A[7,6] * Conv_1_aggregated_features[6,6] ] = 0
 Conv_1_averages_constraint[6,7]: [
   A[0,6] * Conv_1_aggregated_features[6,7]
   + A[1,6] * Conv_1_aggregated_features[6,7]
   + A[2,6] * Conv_1_aggregated_features[6,7]
   + A[3,6] * Conv_1_aggregated_features[6,7]
   + A[4,6] * Conv_1_aggregated_features[6,7]
   + A[5,6] * Conv_1_aggregated_features[6,7]
   - A[6,0] * Conv_0_Relu_ts[0,7] - A[6,1] * Conv_0_Relu_ts[1,7]
   - A[6,2] * Conv_0_Relu_ts[2,7] - A[6,3] * Conv_0_Relu_ts[3,7]
   - A[6,4] * Conv_0_Relu_ts[4,7] - A[6,5] * Conv_0_Relu_ts[5,7]
   - A[6,6] * Conv_0_Relu_ts[6,7]
   + A[6,6] * Conv_1_aggregated_features[6,7]
   - A[6,7] * Conv_0_Relu_ts[7,7]
   + A[7,6] * Conv_1_aggregated_features[6,7] ] = 0
 Conv_1_averages_constraint[6,8]: [
   A[0,6] * Conv_1_aggregated_features[6,8]
   + A[1,6] * Conv_1_aggregated_features[6,8]
   + A[2,6] * Conv_1_aggregated_features[6,8]
   + A[3,6] * Conv_1_aggregated_features[6,8]
   + A[4,6] * Conv_1_aggregated_features[6,8]
   + A[5,6] * Conv_1_aggregated_features[6,8]
   - A[6,0] * Conv_0_Relu_ts[0,8] - A[6,1] * Conv_0_Relu_ts[1,8]
   - A[6,2] * Conv_0_Relu_ts[2,8] - A[6,3] * Conv_0_Relu_ts[3,8]
   - A[6,4] * Conv_0_Relu_ts[4,8] - A[6,5] * Conv_0_Relu_ts[5,8]
   - A[6,6] * Conv_0_Relu_ts[6,8]
   + A[6,6] * Conv_1_aggregated_features[6,8]
   - A[6,7] * Conv_0_Relu_ts[7,8]
   + A[7,6] * Conv_1_aggregated_features[6,8] ] = 0
 Conv_1_averages_constraint[6,9]: [
   A[0,6] * Conv_1_aggregated_features[6,9]
   + A[1,6] * Conv_1_aggregated_features[6,9]
   + A[2,6] * Conv_1_aggregated_features[6,9]
   + A[3,6] * Conv_1_aggregated_features[6,9]
   + A[4,6] * Conv_1_aggregated_features[6,9]
   + A[5,6] * Conv_1_aggregated_features[6,9]
   - A[6,0] * Conv_0_Relu_ts[0,9] - A[6,1] * Conv_0_Relu_ts[1,9]
   - A[6,2] * Conv_0_Relu_ts[2,9] - A[6,3] * Conv_0_Relu_ts[3,9]
   - A[6,4] * Conv_0_Relu_ts[4,9] - A[6,5] * Conv_0_Relu_ts[5,9]
   - A[6,6] * Conv_0_Relu_ts[6,9]
   + A[6,6] * Conv_1_aggregated_features[6,9]
   - A[6,7] * Conv_0_Relu_ts[7,9]
   + A[7,6] * Conv_1_aggregated_features[6,9] ] = 0
 Conv_1_averages_constraint[6,10]: [
   A[0,6] * Conv_1_aggregated_features[6,10]
   + A[1,6] * Conv_1_aggregated_features[6,10]
   + A[2,6] * Conv_1_aggregated_features[6,10]
   + A[3,6] * Conv_1_aggregated_features[6,10]
   + A[4,6] * Conv_1_aggregated_features[6,10]
   + A[5,6] * Conv_1_aggregated_features[6,10]
   - A[6,0] * Conv_0_Relu_ts[0,10] - A[6,1] * Conv_0_Relu_ts[1,10]
   - A[6,2] * Conv_0_Relu_ts[2,10] - A[6,3] * Conv_0_Relu_ts[3,10]
   - A[6,4] * Conv_0_Relu_ts[4,10] - A[6,5] * Conv_0_Relu_ts[5,10]
   - A[6,6] * Conv_0_Relu_ts[6,10]
   + A[6,6] * Conv_1_aggregated_features[6,10]
   - A[6,7] * Conv_0_Relu_ts[7,10]
   + A[7,6] * Conv_1_aggregated_features[6,10] ] = 0
 Conv_1_averages_constraint[6,11]: [
   A[0,6] * Conv_1_aggregated_features[6,11]
   + A[1,6] * Conv_1_aggregated_features[6,11]
   + A[2,6] * Conv_1_aggregated_features[6,11]
   + A[3,6] * Conv_1_aggregated_features[6,11]
   + A[4,6] * Conv_1_aggregated_features[6,11]
   + A[5,6] * Conv_1_aggregated_features[6,11]
   - A[6,0] * Conv_0_Relu_ts[0,11] - A[6,1] * Conv_0_Relu_ts[1,11]
   - A[6,2] * Conv_0_Relu_ts[2,11] - A[6,3] * Conv_0_Relu_ts[3,11]
   - A[6,4] * Conv_0_Relu_ts[4,11] - A[6,5] * Conv_0_Relu_ts[5,11]
   - A[6,6] * Conv_0_Relu_ts[6,11]
   + A[6,6] * Conv_1_aggregated_features[6,11]
   - A[6,7] * Conv_0_Relu_ts[7,11]
   + A[7,6] * Conv_1_aggregated_features[6,11] ] = 0
 Conv_1_averages_constraint[6,12]: [
   A[0,6] * Conv_1_aggregated_features[6,12]
   + A[1,6] * Conv_1_aggregated_features[6,12]
   + A[2,6] * Conv_1_aggregated_features[6,12]
   + A[3,6] * Conv_1_aggregated_features[6,12]
   + A[4,6] * Conv_1_aggregated_features[6,12]
   + A[5,6] * Conv_1_aggregated_features[6,12]
   - A[6,0] * Conv_0_Relu_ts[0,12] - A[6,1] * Conv_0_Relu_ts[1,12]
   - A[6,2] * Conv_0_Relu_ts[2,12] - A[6,3] * Conv_0_Relu_ts[3,12]
   - A[6,4] * Conv_0_Relu_ts[4,12] - A[6,5] * Conv_0_Relu_ts[5,12]
   - A[6,6] * Conv_0_Relu_ts[6,12]
   + A[6,6] * Conv_1_aggregated_features[6,12]
   - A[6,7] * Conv_0_Relu_ts[7,12]
   + A[7,6] * Conv_1_aggregated_features[6,12] ] = 0
 Conv_1_averages_constraint[6,13]: [
   A[0,6] * Conv_1_aggregated_features[6,13]
   + A[1,6] * Conv_1_aggregated_features[6,13]
   + A[2,6] * Conv_1_aggregated_features[6,13]
   + A[3,6] * Conv_1_aggregated_features[6,13]
   + A[4,6] * Conv_1_aggregated_features[6,13]
   + A[5,6] * Conv_1_aggregated_features[6,13]
   - A[6,0] * Conv_0_Relu_ts[0,13] - A[6,1] * Conv_0_Relu_ts[1,13]
   - A[6,2] * Conv_0_Relu_ts[2,13] - A[6,3] * Conv_0_Relu_ts[3,13]
   - A[6,4] * Conv_0_Relu_ts[4,13] - A[6,5] * Conv_0_Relu_ts[5,13]
   - A[6,6] * Conv_0_Relu_ts[6,13]
   + A[6,6] * Conv_1_aggregated_features[6,13]
   - A[6,7] * Conv_0_Relu_ts[7,13]
   + A[7,6] * Conv_1_aggregated_features[6,13] ] = 0
 Conv_1_averages_constraint[6,14]: [
   A[0,6] * Conv_1_aggregated_features[6,14]
   + A[1,6] * Conv_1_aggregated_features[6,14]
   + A[2,6] * Conv_1_aggregated_features[6,14]
   + A[3,6] * Conv_1_aggregated_features[6,14]
   + A[4,6] * Conv_1_aggregated_features[6,14]
   + A[5,6] * Conv_1_aggregated_features[6,14]
   - A[6,0] * Conv_0_Relu_ts[0,14] - A[6,1] * Conv_0_Relu_ts[1,14]
   - A[6,2] * Conv_0_Relu_ts[2,14] - A[6,3] * Conv_0_Relu_ts[3,14]
   - A[6,4] * Conv_0_Relu_ts[4,14] - A[6,5] * Conv_0_Relu_ts[5,14]
   - A[6,6] * Conv_0_Relu_ts[6,14]
   + A[6,6] * Conv_1_aggregated_features[6,14]
   - A[6,7] * Conv_0_Relu_ts[7,14]
   + A[7,6] * Conv_1_aggregated_features[6,14] ] = 0
 Conv_1_averages_constraint[6,15]: [
   A[0,6] * Conv_1_aggregated_features[6,15]
   + A[1,6] * Conv_1_aggregated_features[6,15]
   + A[2,6] * Conv_1_aggregated_features[6,15]
   + A[3,6] * Conv_1_aggregated_features[6,15]
   + A[4,6] * Conv_1_aggregated_features[6,15]
   + A[5,6] * Conv_1_aggregated_features[6,15]
   - A[6,0] * Conv_0_Relu_ts[0,15] - A[6,1] * Conv_0_Relu_ts[1,15]
   - A[6,2] * Conv_0_Relu_ts[2,15] - A[6,3] * Conv_0_Relu_ts[3,15]
   - A[6,4] * Conv_0_Relu_ts[4,15] - A[6,5] * Conv_0_Relu_ts[5,15]
   - A[6,6] * Conv_0_Relu_ts[6,15]
   + A[6,6] * Conv_1_aggregated_features[6,15]
   - A[6,7] * Conv_0_Relu_ts[7,15]
   + A[7,6] * Conv_1_aggregated_features[6,15] ] = 0
 Conv_1_averages_constraint[7,0]: [
   A[0,7] * Conv_1_aggregated_features[7,0]
   + A[1,7] * Conv_1_aggregated_features[7,0]
   + A[2,7] * Conv_1_aggregated_features[7,0]
   + A[3,7] * Conv_1_aggregated_features[7,0]
   + A[4,7] * Conv_1_aggregated_features[7,0]
   + A[5,7] * Conv_1_aggregated_features[7,0]
   + A[6,7] * Conv_1_aggregated_features[7,0]
   - A[7,0] * Conv_0_Relu_ts[0,0] - A[7,1] * Conv_0_Relu_ts[1,0]
   - A[7,2] * Conv_0_Relu_ts[2,0] - A[7,3] * Conv_0_Relu_ts[3,0]
   - A[7,4] * Conv_0_Relu_ts[4,0] - A[7,5] * Conv_0_Relu_ts[5,0]
   - A[7,6] * Conv_0_Relu_ts[6,0] - A[7,7] * Conv_0_Relu_ts[7,0]
   + A[7,7] * Conv_1_aggregated_features[7,0] ] = 0
 Conv_1_averages_constraint[7,1]: [
   A[0,7] * Conv_1_aggregated_features[7,1]
   + A[1,7] * Conv_1_aggregated_features[7,1]
   + A[2,7] * Conv_1_aggregated_features[7,1]
   + A[3,7] * Conv_1_aggregated_features[7,1]
   + A[4,7] * Conv_1_aggregated_features[7,1]
   + A[5,7] * Conv_1_aggregated_features[7,1]
   + A[6,7] * Conv_1_aggregated_features[7,1]
   - A[7,0] * Conv_0_Relu_ts[0,1] - A[7,1] * Conv_0_Relu_ts[1,1]
   - A[7,2] * Conv_0_Relu_ts[2,1] - A[7,3] * Conv_0_Relu_ts[3,1]
   - A[7,4] * Conv_0_Relu_ts[4,1] - A[7,5] * Conv_0_Relu_ts[5,1]
   - A[7,6] * Conv_0_Relu_ts[6,1] - A[7,7] * Conv_0_Relu_ts[7,1]
   + A[7,7] * Conv_1_aggregated_features[7,1] ] = 0
 Conv_1_averages_constraint[7,2]: [
   A[0,7] * Conv_1_aggregated_features[7,2]
   + A[1,7] * Conv_1_aggregated_features[7,2]
   + A[2,7] * Conv_1_aggregated_features[7,2]
   + A[3,7] * Conv_1_aggregated_features[7,2]
   + A[4,7] * Conv_1_aggregated_features[7,2]
   + A[5,7] * Conv_1_aggregated_features[7,2]
   + A[6,7] * Conv_1_aggregated_features[7,2]
   - A[7,0] * Conv_0_Relu_ts[0,2] - A[7,1] * Conv_0_Relu_ts[1,2]
   - A[7,2] * Conv_0_Relu_ts[2,2] - A[7,3] * Conv_0_Relu_ts[3,2]
   - A[7,4] * Conv_0_Relu_ts[4,2] - A[7,5] * Conv_0_Relu_ts[5,2]
   - A[7,6] * Conv_0_Relu_ts[6,2] - A[7,7] * Conv_0_Relu_ts[7,2]
   + A[7,7] * Conv_1_aggregated_features[7,2] ] = 0
 Conv_1_averages_constraint[7,3]: [
   A[0,7] * Conv_1_aggregated_features[7,3]
   + A[1,7] * Conv_1_aggregated_features[7,3]
   + A[2,7] * Conv_1_aggregated_features[7,3]
   + A[3,7] * Conv_1_aggregated_features[7,3]
   + A[4,7] * Conv_1_aggregated_features[7,3]
   + A[5,7] * Conv_1_aggregated_features[7,3]
   + A[6,7] * Conv_1_aggregated_features[7,3]
   - A[7,0] * Conv_0_Relu_ts[0,3] - A[7,1] * Conv_0_Relu_ts[1,3]
   - A[7,2] * Conv_0_Relu_ts[2,3] - A[7,3] * Conv_0_Relu_ts[3,3]
   - A[7,4] * Conv_0_Relu_ts[4,3] - A[7,5] * Conv_0_Relu_ts[5,3]
   - A[7,6] * Conv_0_Relu_ts[6,3] - A[7,7] * Conv_0_Relu_ts[7,3]
   + A[7,7] * Conv_1_aggregated_features[7,3] ] = 0
 Conv_1_averages_constraint[7,4]: [
   A[0,7] * Conv_1_aggregated_features[7,4]
   + A[1,7] * Conv_1_aggregated_features[7,4]
   + A[2,7] * Conv_1_aggregated_features[7,4]
   + A[3,7] * Conv_1_aggregated_features[7,4]
   + A[4,7] * Conv_1_aggregated_features[7,4]
   + A[5,7] * Conv_1_aggregated_features[7,4]
   + A[6,7] * Conv_1_aggregated_features[7,4]
   - A[7,0] * Conv_0_Relu_ts[0,4] - A[7,1] * Conv_0_Relu_ts[1,4]
   - A[7,2] * Conv_0_Relu_ts[2,4] - A[7,3] * Conv_0_Relu_ts[3,4]
   - A[7,4] * Conv_0_Relu_ts[4,4] - A[7,5] * Conv_0_Relu_ts[5,4]
   - A[7,6] * Conv_0_Relu_ts[6,4] - A[7,7] * Conv_0_Relu_ts[7,4]
   + A[7,7] * Conv_1_aggregated_features[7,4] ] = 0
 Conv_1_averages_constraint[7,5]: [
   A[0,7] * Conv_1_aggregated_features[7,5]
   + A[1,7] * Conv_1_aggregated_features[7,5]
   + A[2,7] * Conv_1_aggregated_features[7,5]
   + A[3,7] * Conv_1_aggregated_features[7,5]
   + A[4,7] * Conv_1_aggregated_features[7,5]
   + A[5,7] * Conv_1_aggregated_features[7,5]
   + A[6,7] * Conv_1_aggregated_features[7,5]
   - A[7,0] * Conv_0_Relu_ts[0,5] - A[7,1] * Conv_0_Relu_ts[1,5]
   - A[7,2] * Conv_0_Relu_ts[2,5] - A[7,3] * Conv_0_Relu_ts[3,5]
   - A[7,4] * Conv_0_Relu_ts[4,5] - A[7,5] * Conv_0_Relu_ts[5,5]
   - A[7,6] * Conv_0_Relu_ts[6,5] - A[7,7] * Conv_0_Relu_ts[7,5]
   + A[7,7] * Conv_1_aggregated_features[7,5] ] = 0
 Conv_1_averages_constraint[7,6]: [
   A[0,7] * Conv_1_aggregated_features[7,6]
   + A[1,7] * Conv_1_aggregated_features[7,6]
   + A[2,7] * Conv_1_aggregated_features[7,6]
   + A[3,7] * Conv_1_aggregated_features[7,6]
   + A[4,7] * Conv_1_aggregated_features[7,6]
   + A[5,7] * Conv_1_aggregated_features[7,6]
   + A[6,7] * Conv_1_aggregated_features[7,6]
   - A[7,0] * Conv_0_Relu_ts[0,6] - A[7,1] * Conv_0_Relu_ts[1,6]
   - A[7,2] * Conv_0_Relu_ts[2,6] - A[7,3] * Conv_0_Relu_ts[3,6]
   - A[7,4] * Conv_0_Relu_ts[4,6] - A[7,5] * Conv_0_Relu_ts[5,6]
   - A[7,6] * Conv_0_Relu_ts[6,6] - A[7,7] * Conv_0_Relu_ts[7,6]
   + A[7,7] * Conv_1_aggregated_features[7,6] ] = 0
 Conv_1_averages_constraint[7,7]: [
   A[0,7] * Conv_1_aggregated_features[7,7]
   + A[1,7] * Conv_1_aggregated_features[7,7]
   + A[2,7] * Conv_1_aggregated_features[7,7]
   + A[3,7] * Conv_1_aggregated_features[7,7]
   + A[4,7] * Conv_1_aggregated_features[7,7]
   + A[5,7] * Conv_1_aggregated_features[7,7]
   + A[6,7] * Conv_1_aggregated_features[7,7]
   - A[7,0] * Conv_0_Relu_ts[0,7] - A[7,1] * Conv_0_Relu_ts[1,7]
   - A[7,2] * Conv_0_Relu_ts[2,7] - A[7,3] * Conv_0_Relu_ts[3,7]
   - A[7,4] * Conv_0_Relu_ts[4,7] - A[7,5] * Conv_0_Relu_ts[5,7]
   - A[7,6] * Conv_0_Relu_ts[6,7] - A[7,7] * Conv_0_Relu_ts[7,7]
   + A[7,7] * Conv_1_aggregated_features[7,7] ] = 0
 Conv_1_averages_constraint[7,8]: [
   A[0,7] * Conv_1_aggregated_features[7,8]
   + A[1,7] * Conv_1_aggregated_features[7,8]
   + A[2,7] * Conv_1_aggregated_features[7,8]
   + A[3,7] * Conv_1_aggregated_features[7,8]
   + A[4,7] * Conv_1_aggregated_features[7,8]
   + A[5,7] * Conv_1_aggregated_features[7,8]
   + A[6,7] * Conv_1_aggregated_features[7,8]
   - A[7,0] * Conv_0_Relu_ts[0,8] - A[7,1] * Conv_0_Relu_ts[1,8]
   - A[7,2] * Conv_0_Relu_ts[2,8] - A[7,3] * Conv_0_Relu_ts[3,8]
   - A[7,4] * Conv_0_Relu_ts[4,8] - A[7,5] * Conv_0_Relu_ts[5,8]
   - A[7,6] * Conv_0_Relu_ts[6,8] - A[7,7] * Conv_0_Relu_ts[7,8]
   + A[7,7] * Conv_1_aggregated_features[7,8] ] = 0
 Conv_1_averages_constraint[7,9]: [
   A[0,7] * Conv_1_aggregated_features[7,9]
   + A[1,7] * Conv_1_aggregated_features[7,9]
   + A[2,7] * Conv_1_aggregated_features[7,9]
   + A[3,7] * Conv_1_aggregated_features[7,9]
   + A[4,7] * Conv_1_aggregated_features[7,9]
   + A[5,7] * Conv_1_aggregated_features[7,9]
   + A[6,7] * Conv_1_aggregated_features[7,9]
   - A[7,0] * Conv_0_Relu_ts[0,9] - A[7,1] * Conv_0_Relu_ts[1,9]
   - A[7,2] * Conv_0_Relu_ts[2,9] - A[7,3] * Conv_0_Relu_ts[3,9]
   - A[7,4] * Conv_0_Relu_ts[4,9] - A[7,5] * Conv_0_Relu_ts[5,9]
   - A[7,6] * Conv_0_Relu_ts[6,9] - A[7,7] * Conv_0_Relu_ts[7,9]
   + A[7,7] * Conv_1_aggregated_features[7,9] ] = 0
 Conv_1_averages_constraint[7,10]: [
   A[0,7] * Conv_1_aggregated_features[7,10]
   + A[1,7] * Conv_1_aggregated_features[7,10]
   + A[2,7] * Conv_1_aggregated_features[7,10]
   + A[3,7] * Conv_1_aggregated_features[7,10]
   + A[4,7] * Conv_1_aggregated_features[7,10]
   + A[5,7] * Conv_1_aggregated_features[7,10]
   + A[6,7] * Conv_1_aggregated_features[7,10]
   - A[7,0] * Conv_0_Relu_ts[0,10] - A[7,1] * Conv_0_Relu_ts[1,10]
   - A[7,2] * Conv_0_Relu_ts[2,10] - A[7,3] * Conv_0_Relu_ts[3,10]
   - A[7,4] * Conv_0_Relu_ts[4,10] - A[7,5] * Conv_0_Relu_ts[5,10]
   - A[7,6] * Conv_0_Relu_ts[6,10] - A[7,7] * Conv_0_Relu_ts[7,10]
   + A[7,7] * Conv_1_aggregated_features[7,10] ] = 0
 Conv_1_averages_constraint[7,11]: [
   A[0,7] * Conv_1_aggregated_features[7,11]
   + A[1,7] * Conv_1_aggregated_features[7,11]
   + A[2,7] * Conv_1_aggregated_features[7,11]
   + A[3,7] * Conv_1_aggregated_features[7,11]
   + A[4,7] * Conv_1_aggregated_features[7,11]
   + A[5,7] * Conv_1_aggregated_features[7,11]
   + A[6,7] * Conv_1_aggregated_features[7,11]
   - A[7,0] * Conv_0_Relu_ts[0,11] - A[7,1] * Conv_0_Relu_ts[1,11]
   - A[7,2] * Conv_0_Relu_ts[2,11] - A[7,3] * Conv_0_Relu_ts[3,11]
   - A[7,4] * Conv_0_Relu_ts[4,11] - A[7,5] * Conv_0_Relu_ts[5,11]
   - A[7,6] * Conv_0_Relu_ts[6,11] - A[7,7] * Conv_0_Relu_ts[7,11]
   + A[7,7] * Conv_1_aggregated_features[7,11] ] = 0
 Conv_1_averages_constraint[7,12]: [
   A[0,7] * Conv_1_aggregated_features[7,12]
   + A[1,7] * Conv_1_aggregated_features[7,12]
   + A[2,7] * Conv_1_aggregated_features[7,12]
   + A[3,7] * Conv_1_aggregated_features[7,12]
   + A[4,7] * Conv_1_aggregated_features[7,12]
   + A[5,7] * Conv_1_aggregated_features[7,12]
   + A[6,7] * Conv_1_aggregated_features[7,12]
   - A[7,0] * Conv_0_Relu_ts[0,12] - A[7,1] * Conv_0_Relu_ts[1,12]
   - A[7,2] * Conv_0_Relu_ts[2,12] - A[7,3] * Conv_0_Relu_ts[3,12]
   - A[7,4] * Conv_0_Relu_ts[4,12] - A[7,5] * Conv_0_Relu_ts[5,12]
   - A[7,6] * Conv_0_Relu_ts[6,12] - A[7,7] * Conv_0_Relu_ts[7,12]
   + A[7,7] * Conv_1_aggregated_features[7,12] ] = 0
 Conv_1_averages_constraint[7,13]: [
   A[0,7] * Conv_1_aggregated_features[7,13]
   + A[1,7] * Conv_1_aggregated_features[7,13]
   + A[2,7] * Conv_1_aggregated_features[7,13]
   + A[3,7] * Conv_1_aggregated_features[7,13]
   + A[4,7] * Conv_1_aggregated_features[7,13]
   + A[5,7] * Conv_1_aggregated_features[7,13]
   + A[6,7] * Conv_1_aggregated_features[7,13]
   - A[7,0] * Conv_0_Relu_ts[0,13] - A[7,1] * Conv_0_Relu_ts[1,13]
   - A[7,2] * Conv_0_Relu_ts[2,13] - A[7,3] * Conv_0_Relu_ts[3,13]
   - A[7,4] * Conv_0_Relu_ts[4,13] - A[7,5] * Conv_0_Relu_ts[5,13]
   - A[7,6] * Conv_0_Relu_ts[6,13] - A[7,7] * Conv_0_Relu_ts[7,13]
   + A[7,7] * Conv_1_aggregated_features[7,13] ] = 0
 Conv_1_averages_constraint[7,14]: [
   A[0,7] * Conv_1_aggregated_features[7,14]
   + A[1,7] * Conv_1_aggregated_features[7,14]
   + A[2,7] * Conv_1_aggregated_features[7,14]
   + A[3,7] * Conv_1_aggregated_features[7,14]
   + A[4,7] * Conv_1_aggregated_features[7,14]
   + A[5,7] * Conv_1_aggregated_features[7,14]
   + A[6,7] * Conv_1_aggregated_features[7,14]
   - A[7,0] * Conv_0_Relu_ts[0,14] - A[7,1] * Conv_0_Relu_ts[1,14]
   - A[7,2] * Conv_0_Relu_ts[2,14] - A[7,3] * Conv_0_Relu_ts[3,14]
   - A[7,4] * Conv_0_Relu_ts[4,14] - A[7,5] * Conv_0_Relu_ts[5,14]
   - A[7,6] * Conv_0_Relu_ts[6,14] - A[7,7] * Conv_0_Relu_ts[7,14]
   + A[7,7] * Conv_1_aggregated_features[7,14] ] = 0
 Conv_1_averages_constraint[7,15]: [
   A[0,7] * Conv_1_aggregated_features[7,15]
   + A[1,7] * Conv_1_aggregated_features[7,15]
   + A[2,7] * Conv_1_aggregated_features[7,15]
   + A[3,7] * Conv_1_aggregated_features[7,15]
   + A[4,7] * Conv_1_aggregated_features[7,15]
   + A[5,7] * Conv_1_aggregated_features[7,15]
   + A[6,7] * Conv_1_aggregated_features[7,15]
   - A[7,0] * Conv_0_Relu_ts[0,15] - A[7,1] * Conv_0_Relu_ts[1,15]
   - A[7,2] * Conv_0_Relu_ts[2,15] - A[7,3] * Conv_0_Relu_ts[3,15]
   - A[7,4] * Conv_0_Relu_ts[4,15] - A[7,5] * Conv_0_Relu_ts[5,15]
   - A[7,6] * Conv_0_Relu_ts[6,15] - A[7,7] * Conv_0_Relu_ts[7,15]
   + A[7,7] * Conv_1_aggregated_features[7,15] ] = 0
 Conv_2_averages_constraint[0,0]: [ - A[0,0] * Conv_1_Relu_ts[0,0]
   + A[0,0] * Conv_2_aggregated_features[0,0]
   - A[0,1] * Conv_1_Relu_ts[1,0] - A[0,2] * Conv_1_Relu_ts[2,0]
   - A[0,3] * Conv_1_Relu_ts[3,0] - A[0,4] * Conv_1_Relu_ts[4,0]
   - A[0,5] * Conv_1_Relu_ts[5,0] - A[0,6] * Conv_1_Relu_ts[6,0]
   - A[0,7] * Conv_1_Relu_ts[7,0]
   + A[1,0] * Conv_2_aggregated_features[0,0]
   + A[2,0] * Conv_2_aggregated_features[0,0]
   + A[3,0] * Conv_2_aggregated_features[0,0]
   + A[4,0] * Conv_2_aggregated_features[0,0]
   + A[5,0] * Conv_2_aggregated_features[0,0]
   + A[6,0] * Conv_2_aggregated_features[0,0]
   + A[7,0] * Conv_2_aggregated_features[0,0] ] = 0
 Conv_2_averages_constraint[0,1]: [ - A[0,0] * Conv_1_Relu_ts[0,1]
   + A[0,0] * Conv_2_aggregated_features[0,1]
   - A[0,1] * Conv_1_Relu_ts[1,1] - A[0,2] * Conv_1_Relu_ts[2,1]
   - A[0,3] * Conv_1_Relu_ts[3,1] - A[0,4] * Conv_1_Relu_ts[4,1]
   - A[0,5] * Conv_1_Relu_ts[5,1] - A[0,6] * Conv_1_Relu_ts[6,1]
   - A[0,7] * Conv_1_Relu_ts[7,1]
   + A[1,0] * Conv_2_aggregated_features[0,1]
   + A[2,0] * Conv_2_aggregated_features[0,1]
   + A[3,0] * Conv_2_aggregated_features[0,1]
   + A[4,0] * Conv_2_aggregated_features[0,1]
   + A[5,0] * Conv_2_aggregated_features[0,1]
   + A[6,0] * Conv_2_aggregated_features[0,1]
   + A[7,0] * Conv_2_aggregated_features[0,1] ] = 0
 Conv_2_averages_constraint[0,2]: [ - A[0,0] * Conv_1_Relu_ts[0,2]
   + A[0,0] * Conv_2_aggregated_features[0,2]
   - A[0,1] * Conv_1_Relu_ts[1,2] - A[0,2] * Conv_1_Relu_ts[2,2]
   - A[0,3] * Conv_1_Relu_ts[3,2] - A[0,4] * Conv_1_Relu_ts[4,2]
   - A[0,5] * Conv_1_Relu_ts[5,2] - A[0,6] * Conv_1_Relu_ts[6,2]
   - A[0,7] * Conv_1_Relu_ts[7,2]
   + A[1,0] * Conv_2_aggregated_features[0,2]
   + A[2,0] * Conv_2_aggregated_features[0,2]
   + A[3,0] * Conv_2_aggregated_features[0,2]
   + A[4,0] * Conv_2_aggregated_features[0,2]
   + A[5,0] * Conv_2_aggregated_features[0,2]
   + A[6,0] * Conv_2_aggregated_features[0,2]
   + A[7,0] * Conv_2_aggregated_features[0,2] ] = 0
 Conv_2_averages_constraint[0,3]: [ - A[0,0] * Conv_1_Relu_ts[0,3]
   + A[0,0] * Conv_2_aggregated_features[0,3]
   - A[0,1] * Conv_1_Relu_ts[1,3] - A[0,2] * Conv_1_Relu_ts[2,3]
   - A[0,3] * Conv_1_Relu_ts[3,3] - A[0,4] * Conv_1_Relu_ts[4,3]
   - A[0,5] * Conv_1_Relu_ts[5,3] - A[0,6] * Conv_1_Relu_ts[6,3]
   - A[0,7] * Conv_1_Relu_ts[7,3]
   + A[1,0] * Conv_2_aggregated_features[0,3]
   + A[2,0] * Conv_2_aggregated_features[0,3]
   + A[3,0] * Conv_2_aggregated_features[0,3]
   + A[4,0] * Conv_2_aggregated_features[0,3]
   + A[5,0] * Conv_2_aggregated_features[0,3]
   + A[6,0] * Conv_2_aggregated_features[0,3]
   + A[7,0] * Conv_2_aggregated_features[0,3] ] = 0
 Conv_2_averages_constraint[0,4]: [ - A[0,0] * Conv_1_Relu_ts[0,4]
   + A[0,0] * Conv_2_aggregated_features[0,4]
   - A[0,1] * Conv_1_Relu_ts[1,4] - A[0,2] * Conv_1_Relu_ts[2,4]
   - A[0,3] * Conv_1_Relu_ts[3,4] - A[0,4] * Conv_1_Relu_ts[4,4]
   - A[0,5] * Conv_1_Relu_ts[5,4] - A[0,6] * Conv_1_Relu_ts[6,4]
   - A[0,7] * Conv_1_Relu_ts[7,4]
   + A[1,0] * Conv_2_aggregated_features[0,4]
   + A[2,0] * Conv_2_aggregated_features[0,4]
   + A[3,0] * Conv_2_aggregated_features[0,4]
   + A[4,0] * Conv_2_aggregated_features[0,4]
   + A[5,0] * Conv_2_aggregated_features[0,4]
   + A[6,0] * Conv_2_aggregated_features[0,4]
   + A[7,0] * Conv_2_aggregated_features[0,4] ] = 0
 Conv_2_averages_constraint[0,5]: [ - A[0,0] * Conv_1_Relu_ts[0,5]
   + A[0,0] * Conv_2_aggregated_features[0,5]
   - A[0,1] * Conv_1_Relu_ts[1,5] - A[0,2] * Conv_1_Relu_ts[2,5]
   - A[0,3] * Conv_1_Relu_ts[3,5] - A[0,4] * Conv_1_Relu_ts[4,5]
   - A[0,5] * Conv_1_Relu_ts[5,5] - A[0,6] * Conv_1_Relu_ts[6,5]
   - A[0,7] * Conv_1_Relu_ts[7,5]
   + A[1,0] * Conv_2_aggregated_features[0,5]
   + A[2,0] * Conv_2_aggregated_features[0,5]
   + A[3,0] * Conv_2_aggregated_features[0,5]
   + A[4,0] * Conv_2_aggregated_features[0,5]
   + A[5,0] * Conv_2_aggregated_features[0,5]
   + A[6,0] * Conv_2_aggregated_features[0,5]
   + A[7,0] * Conv_2_aggregated_features[0,5] ] = 0
 Conv_2_averages_constraint[0,6]: [ - A[0,0] * Conv_1_Relu_ts[0,6]
   + A[0,0] * Conv_2_aggregated_features[0,6]
   - A[0,1] * Conv_1_Relu_ts[1,6] - A[0,2] * Conv_1_Relu_ts[2,6]
   - A[0,3] * Conv_1_Relu_ts[3,6] - A[0,4] * Conv_1_Relu_ts[4,6]
   - A[0,5] * Conv_1_Relu_ts[5,6] - A[0,6] * Conv_1_Relu_ts[6,6]
   - A[0,7] * Conv_1_Relu_ts[7,6]
   + A[1,0] * Conv_2_aggregated_features[0,6]
   + A[2,0] * Conv_2_aggregated_features[0,6]
   + A[3,0] * Conv_2_aggregated_features[0,6]
   + A[4,0] * Conv_2_aggregated_features[0,6]
   + A[5,0] * Conv_2_aggregated_features[0,6]
   + A[6,0] * Conv_2_aggregated_features[0,6]
   + A[7,0] * Conv_2_aggregated_features[0,6] ] = 0
 Conv_2_averages_constraint[0,7]: [ - A[0,0] * Conv_1_Relu_ts[0,7]
   + A[0,0] * Conv_2_aggregated_features[0,7]
   - A[0,1] * Conv_1_Relu_ts[1,7] - A[0,2] * Conv_1_Relu_ts[2,7]
   - A[0,3] * Conv_1_Relu_ts[3,7] - A[0,4] * Conv_1_Relu_ts[4,7]
   - A[0,5] * Conv_1_Relu_ts[5,7] - A[0,6] * Conv_1_Relu_ts[6,7]
   - A[0,7] * Conv_1_Relu_ts[7,7]
   + A[1,0] * Conv_2_aggregated_features[0,7]
   + A[2,0] * Conv_2_aggregated_features[0,7]
   + A[3,0] * Conv_2_aggregated_features[0,7]
   + A[4,0] * Conv_2_aggregated_features[0,7]
   + A[5,0] * Conv_2_aggregated_features[0,7]
   + A[6,0] * Conv_2_aggregated_features[0,7]
   + A[7,0] * Conv_2_aggregated_features[0,7] ] = 0
 Conv_2_averages_constraint[0,8]: [ - A[0,0] * Conv_1_Relu_ts[0,8]
   + A[0,0] * Conv_2_aggregated_features[0,8]
   - A[0,1] * Conv_1_Relu_ts[1,8] - A[0,2] * Conv_1_Relu_ts[2,8]
   - A[0,3] * Conv_1_Relu_ts[3,8] - A[0,4] * Conv_1_Relu_ts[4,8]
   - A[0,5] * Conv_1_Relu_ts[5,8] - A[0,6] * Conv_1_Relu_ts[6,8]
   - A[0,7] * Conv_1_Relu_ts[7,8]
   + A[1,0] * Conv_2_aggregated_features[0,8]
   + A[2,0] * Conv_2_aggregated_features[0,8]
   + A[3,0] * Conv_2_aggregated_features[0,8]
   + A[4,0] * Conv_2_aggregated_features[0,8]
   + A[5,0] * Conv_2_aggregated_features[0,8]
   + A[6,0] * Conv_2_aggregated_features[0,8]
   + A[7,0] * Conv_2_aggregated_features[0,8] ] = 0
 Conv_2_averages_constraint[0,9]: [ - A[0,0] * Conv_1_Relu_ts[0,9]
   + A[0,0] * Conv_2_aggregated_features[0,9]
   - A[0,1] * Conv_1_Relu_ts[1,9] - A[0,2] * Conv_1_Relu_ts[2,9]
   - A[0,3] * Conv_1_Relu_ts[3,9] - A[0,4] * Conv_1_Relu_ts[4,9]
   - A[0,5] * Conv_1_Relu_ts[5,9] - A[0,6] * Conv_1_Relu_ts[6,9]
   - A[0,7] * Conv_1_Relu_ts[7,9]
   + A[1,0] * Conv_2_aggregated_features[0,9]
   + A[2,0] * Conv_2_aggregated_features[0,9]
   + A[3,0] * Conv_2_aggregated_features[0,9]
   + A[4,0] * Conv_2_aggregated_features[0,9]
   + A[5,0] * Conv_2_aggregated_features[0,9]
   + A[6,0] * Conv_2_aggregated_features[0,9]
   + A[7,0] * Conv_2_aggregated_features[0,9] ] = 0
 Conv_2_averages_constraint[0,10]: [ - A[0,0] * Conv_1_Relu_ts[0,10]
   + A[0,0] * Conv_2_aggregated_features[0,10]
   - A[0,1] * Conv_1_Relu_ts[1,10] - A[0,2] * Conv_1_Relu_ts[2,10]
   - A[0,3] * Conv_1_Relu_ts[3,10] - A[0,4] * Conv_1_Relu_ts[4,10]
   - A[0,5] * Conv_1_Relu_ts[5,10] - A[0,6] * Conv_1_Relu_ts[6,10]
   - A[0,7] * Conv_1_Relu_ts[7,10]
   + A[1,0] * Conv_2_aggregated_features[0,10]
   + A[2,0] * Conv_2_aggregated_features[0,10]
   + A[3,0] * Conv_2_aggregated_features[0,10]
   + A[4,0] * Conv_2_aggregated_features[0,10]
   + A[5,0] * Conv_2_aggregated_features[0,10]
   + A[6,0] * Conv_2_aggregated_features[0,10]
   + A[7,0] * Conv_2_aggregated_features[0,10] ] = 0
 Conv_2_averages_constraint[0,11]: [ - A[0,0] * Conv_1_Relu_ts[0,11]
   + A[0,0] * Conv_2_aggregated_features[0,11]
   - A[0,1] * Conv_1_Relu_ts[1,11] - A[0,2] * Conv_1_Relu_ts[2,11]
   - A[0,3] * Conv_1_Relu_ts[3,11] - A[0,4] * Conv_1_Relu_ts[4,11]
   - A[0,5] * Conv_1_Relu_ts[5,11] - A[0,6] * Conv_1_Relu_ts[6,11]
   - A[0,7] * Conv_1_Relu_ts[7,11]
   + A[1,0] * Conv_2_aggregated_features[0,11]
   + A[2,0] * Conv_2_aggregated_features[0,11]
   + A[3,0] * Conv_2_aggregated_features[0,11]
   + A[4,0] * Conv_2_aggregated_features[0,11]
   + A[5,0] * Conv_2_aggregated_features[0,11]
   + A[6,0] * Conv_2_aggregated_features[0,11]
   + A[7,0] * Conv_2_aggregated_features[0,11] ] = 0
 Conv_2_averages_constraint[0,12]: [ - A[0,0] * Conv_1_Relu_ts[0,12]
   + A[0,0] * Conv_2_aggregated_features[0,12]
   - A[0,1] * Conv_1_Relu_ts[1,12] - A[0,2] * Conv_1_Relu_ts[2,12]
   - A[0,3] * Conv_1_Relu_ts[3,12] - A[0,4] * Conv_1_Relu_ts[4,12]
   - A[0,5] * Conv_1_Relu_ts[5,12] - A[0,6] * Conv_1_Relu_ts[6,12]
   - A[0,7] * Conv_1_Relu_ts[7,12]
   + A[1,0] * Conv_2_aggregated_features[0,12]
   + A[2,0] * Conv_2_aggregated_features[0,12]
   + A[3,0] * Conv_2_aggregated_features[0,12]
   + A[4,0] * Conv_2_aggregated_features[0,12]
   + A[5,0] * Conv_2_aggregated_features[0,12]
   + A[6,0] * Conv_2_aggregated_features[0,12]
   + A[7,0] * Conv_2_aggregated_features[0,12] ] = 0
 Conv_2_averages_constraint[0,13]: [ - A[0,0] * Conv_1_Relu_ts[0,13]
   + A[0,0] * Conv_2_aggregated_features[0,13]
   - A[0,1] * Conv_1_Relu_ts[1,13] - A[0,2] * Conv_1_Relu_ts[2,13]
   - A[0,3] * Conv_1_Relu_ts[3,13] - A[0,4] * Conv_1_Relu_ts[4,13]
   - A[0,5] * Conv_1_Relu_ts[5,13] - A[0,6] * Conv_1_Relu_ts[6,13]
   - A[0,7] * Conv_1_Relu_ts[7,13]
   + A[1,0] * Conv_2_aggregated_features[0,13]
   + A[2,0] * Conv_2_aggregated_features[0,13]
   + A[3,0] * Conv_2_aggregated_features[0,13]
   + A[4,0] * Conv_2_aggregated_features[0,13]
   + A[5,0] * Conv_2_aggregated_features[0,13]
   + A[6,0] * Conv_2_aggregated_features[0,13]
   + A[7,0] * Conv_2_aggregated_features[0,13] ] = 0
 Conv_2_averages_constraint[0,14]: [ - A[0,0] * Conv_1_Relu_ts[0,14]
   + A[0,0] * Conv_2_aggregated_features[0,14]
   - A[0,1] * Conv_1_Relu_ts[1,14] - A[0,2] * Conv_1_Relu_ts[2,14]
   - A[0,3] * Conv_1_Relu_ts[3,14] - A[0,4] * Conv_1_Relu_ts[4,14]
   - A[0,5] * Conv_1_Relu_ts[5,14] - A[0,6] * Conv_1_Relu_ts[6,14]
   - A[0,7] * Conv_1_Relu_ts[7,14]
   + A[1,0] * Conv_2_aggregated_features[0,14]
   + A[2,0] * Conv_2_aggregated_features[0,14]
   + A[3,0] * Conv_2_aggregated_features[0,14]
   + A[4,0] * Conv_2_aggregated_features[0,14]
   + A[5,0] * Conv_2_aggregated_features[0,14]
   + A[6,0] * Conv_2_aggregated_features[0,14]
   + A[7,0] * Conv_2_aggregated_features[0,14] ] = 0
 Conv_2_averages_constraint[0,15]: [ - A[0,0] * Conv_1_Relu_ts[0,15]
   + A[0,0] * Conv_2_aggregated_features[0,15]
   - A[0,1] * Conv_1_Relu_ts[1,15] - A[0,2] * Conv_1_Relu_ts[2,15]
   - A[0,3] * Conv_1_Relu_ts[3,15] - A[0,4] * Conv_1_Relu_ts[4,15]
   - A[0,5] * Conv_1_Relu_ts[5,15] - A[0,6] * Conv_1_Relu_ts[6,15]
   - A[0,7] * Conv_1_Relu_ts[7,15]
   + A[1,0] * Conv_2_aggregated_features[0,15]
   + A[2,0] * Conv_2_aggregated_features[0,15]
   + A[3,0] * Conv_2_aggregated_features[0,15]
   + A[4,0] * Conv_2_aggregated_features[0,15]
   + A[5,0] * Conv_2_aggregated_features[0,15]
   + A[6,0] * Conv_2_aggregated_features[0,15]
   + A[7,0] * Conv_2_aggregated_features[0,15] ] = 0
 Conv_2_averages_constraint[1,0]: [
   A[0,1] * Conv_2_aggregated_features[1,0] - A[1,0] * Conv_1_Relu_ts[0,0]
   - A[1,1] * Conv_1_Relu_ts[1,0]
   + A[1,1] * Conv_2_aggregated_features[1,0]
   - A[1,2] * Conv_1_Relu_ts[2,0] - A[1,3] * Conv_1_Relu_ts[3,0]
   - A[1,4] * Conv_1_Relu_ts[4,0] - A[1,5] * Conv_1_Relu_ts[5,0]
   - A[1,6] * Conv_1_Relu_ts[6,0] - A[1,7] * Conv_1_Relu_ts[7,0]
   + A[2,1] * Conv_2_aggregated_features[1,0]
   + A[3,1] * Conv_2_aggregated_features[1,0]
   + A[4,1] * Conv_2_aggregated_features[1,0]
   + A[5,1] * Conv_2_aggregated_features[1,0]
   + A[6,1] * Conv_2_aggregated_features[1,0]
   + A[7,1] * Conv_2_aggregated_features[1,0] ] = 0
 Conv_2_averages_constraint[1,1]: [
   A[0,1] * Conv_2_aggregated_features[1,1] - A[1,0] * Conv_1_Relu_ts[0,1]
   - A[1,1] * Conv_1_Relu_ts[1,1]
   + A[1,1] * Conv_2_aggregated_features[1,1]
   - A[1,2] * Conv_1_Relu_ts[2,1] - A[1,3] * Conv_1_Relu_ts[3,1]
   - A[1,4] * Conv_1_Relu_ts[4,1] - A[1,5] * Conv_1_Relu_ts[5,1]
   - A[1,6] * Conv_1_Relu_ts[6,1] - A[1,7] * Conv_1_Relu_ts[7,1]
   + A[2,1] * Conv_2_aggregated_features[1,1]
   + A[3,1] * Conv_2_aggregated_features[1,1]
   + A[4,1] * Conv_2_aggregated_features[1,1]
   + A[5,1] * Conv_2_aggregated_features[1,1]
   + A[6,1] * Conv_2_aggregated_features[1,1]
   + A[7,1] * Conv_2_aggregated_features[1,1] ] = 0
 Conv_2_averages_constraint[1,2]: [
   A[0,1] * Conv_2_aggregated_features[1,2] - A[1,0] * Conv_1_Relu_ts[0,2]
   - A[1,1] * Conv_1_Relu_ts[1,2]
   + A[1,1] * Conv_2_aggregated_features[1,2]
   - A[1,2] * Conv_1_Relu_ts[2,2] - A[1,3] * Conv_1_Relu_ts[3,2]
   - A[1,4] * Conv_1_Relu_ts[4,2] - A[1,5] * Conv_1_Relu_ts[5,2]
   - A[1,6] * Conv_1_Relu_ts[6,2] - A[1,7] * Conv_1_Relu_ts[7,2]
   + A[2,1] * Conv_2_aggregated_features[1,2]
   + A[3,1] * Conv_2_aggregated_features[1,2]
   + A[4,1] * Conv_2_aggregated_features[1,2]
   + A[5,1] * Conv_2_aggregated_features[1,2]
   + A[6,1] * Conv_2_aggregated_features[1,2]
   + A[7,1] * Conv_2_aggregated_features[1,2] ] = 0
 Conv_2_averages_constraint[1,3]: [
   A[0,1] * Conv_2_aggregated_features[1,3] - A[1,0] * Conv_1_Relu_ts[0,3]
   - A[1,1] * Conv_1_Relu_ts[1,3]
   + A[1,1] * Conv_2_aggregated_features[1,3]
   - A[1,2] * Conv_1_Relu_ts[2,3] - A[1,3] * Conv_1_Relu_ts[3,3]
   - A[1,4] * Conv_1_Relu_ts[4,3] - A[1,5] * Conv_1_Relu_ts[5,3]
   - A[1,6] * Conv_1_Relu_ts[6,3] - A[1,7] * Conv_1_Relu_ts[7,3]
   + A[2,1] * Conv_2_aggregated_features[1,3]
   + A[3,1] * Conv_2_aggregated_features[1,3]
   + A[4,1] * Conv_2_aggregated_features[1,3]
   + A[5,1] * Conv_2_aggregated_features[1,3]
   + A[6,1] * Conv_2_aggregated_features[1,3]
   + A[7,1] * Conv_2_aggregated_features[1,3] ] = 0
 Conv_2_averages_constraint[1,4]: [
   A[0,1] * Conv_2_aggregated_features[1,4] - A[1,0] * Conv_1_Relu_ts[0,4]
   - A[1,1] * Conv_1_Relu_ts[1,4]
   + A[1,1] * Conv_2_aggregated_features[1,4]
   - A[1,2] * Conv_1_Relu_ts[2,4] - A[1,3] * Conv_1_Relu_ts[3,4]
   - A[1,4] * Conv_1_Relu_ts[4,4] - A[1,5] * Conv_1_Relu_ts[5,4]
   - A[1,6] * Conv_1_Relu_ts[6,4] - A[1,7] * Conv_1_Relu_ts[7,4]
   + A[2,1] * Conv_2_aggregated_features[1,4]
   + A[3,1] * Conv_2_aggregated_features[1,4]
   + A[4,1] * Conv_2_aggregated_features[1,4]
   + A[5,1] * Conv_2_aggregated_features[1,4]
   + A[6,1] * Conv_2_aggregated_features[1,4]
   + A[7,1] * Conv_2_aggregated_features[1,4] ] = 0
 Conv_2_averages_constraint[1,5]: [
   A[0,1] * Conv_2_aggregated_features[1,5] - A[1,0] * Conv_1_Relu_ts[0,5]
   - A[1,1] * Conv_1_Relu_ts[1,5]
   + A[1,1] * Conv_2_aggregated_features[1,5]
   - A[1,2] * Conv_1_Relu_ts[2,5] - A[1,3] * Conv_1_Relu_ts[3,5]
   - A[1,4] * Conv_1_Relu_ts[4,5] - A[1,5] * Conv_1_Relu_ts[5,5]
   - A[1,6] * Conv_1_Relu_ts[6,5] - A[1,7] * Conv_1_Relu_ts[7,5]
   + A[2,1] * Conv_2_aggregated_features[1,5]
   + A[3,1] * Conv_2_aggregated_features[1,5]
   + A[4,1] * Conv_2_aggregated_features[1,5]
   + A[5,1] * Conv_2_aggregated_features[1,5]
   + A[6,1] * Conv_2_aggregated_features[1,5]
   + A[7,1] * Conv_2_aggregated_features[1,5] ] = 0
 Conv_2_averages_constraint[1,6]: [
   A[0,1] * Conv_2_aggregated_features[1,6] - A[1,0] * Conv_1_Relu_ts[0,6]
   - A[1,1] * Conv_1_Relu_ts[1,6]
   + A[1,1] * Conv_2_aggregated_features[1,6]
   - A[1,2] * Conv_1_Relu_ts[2,6] - A[1,3] * Conv_1_Relu_ts[3,6]
   - A[1,4] * Conv_1_Relu_ts[4,6] - A[1,5] * Conv_1_Relu_ts[5,6]
   - A[1,6] * Conv_1_Relu_ts[6,6] - A[1,7] * Conv_1_Relu_ts[7,6]
   + A[2,1] * Conv_2_aggregated_features[1,6]
   + A[3,1] * Conv_2_aggregated_features[1,6]
   + A[4,1] * Conv_2_aggregated_features[1,6]
   + A[5,1] * Conv_2_aggregated_features[1,6]
   + A[6,1] * Conv_2_aggregated_features[1,6]
   + A[7,1] * Conv_2_aggregated_features[1,6] ] = 0
 Conv_2_averages_constraint[1,7]: [
   A[0,1] * Conv_2_aggregated_features[1,7] - A[1,0] * Conv_1_Relu_ts[0,7]
   - A[1,1] * Conv_1_Relu_ts[1,7]
   + A[1,1] * Conv_2_aggregated_features[1,7]
   - A[1,2] * Conv_1_Relu_ts[2,7] - A[1,3] * Conv_1_Relu_ts[3,7]
   - A[1,4] * Conv_1_Relu_ts[4,7] - A[1,5] * Conv_1_Relu_ts[5,7]
   - A[1,6] * Conv_1_Relu_ts[6,7] - A[1,7] * Conv_1_Relu_ts[7,7]
   + A[2,1] * Conv_2_aggregated_features[1,7]
   + A[3,1] * Conv_2_aggregated_features[1,7]
   + A[4,1] * Conv_2_aggregated_features[1,7]
   + A[5,1] * Conv_2_aggregated_features[1,7]
   + A[6,1] * Conv_2_aggregated_features[1,7]
   + A[7,1] * Conv_2_aggregated_features[1,7] ] = 0
 Conv_2_averages_constraint[1,8]: [
   A[0,1] * Conv_2_aggregated_features[1,8] - A[1,0] * Conv_1_Relu_ts[0,8]
   - A[1,1] * Conv_1_Relu_ts[1,8]
   + A[1,1] * Conv_2_aggregated_features[1,8]
   - A[1,2] * Conv_1_Relu_ts[2,8] - A[1,3] * Conv_1_Relu_ts[3,8]
   - A[1,4] * Conv_1_Relu_ts[4,8] - A[1,5] * Conv_1_Relu_ts[5,8]
   - A[1,6] * Conv_1_Relu_ts[6,8] - A[1,7] * Conv_1_Relu_ts[7,8]
   + A[2,1] * Conv_2_aggregated_features[1,8]
   + A[3,1] * Conv_2_aggregated_features[1,8]
   + A[4,1] * Conv_2_aggregated_features[1,8]
   + A[5,1] * Conv_2_aggregated_features[1,8]
   + A[6,1] * Conv_2_aggregated_features[1,8]
   + A[7,1] * Conv_2_aggregated_features[1,8] ] = 0
 Conv_2_averages_constraint[1,9]: [
   A[0,1] * Conv_2_aggregated_features[1,9] - A[1,0] * Conv_1_Relu_ts[0,9]
   - A[1,1] * Conv_1_Relu_ts[1,9]
   + A[1,1] * Conv_2_aggregated_features[1,9]
   - A[1,2] * Conv_1_Relu_ts[2,9] - A[1,3] * Conv_1_Relu_ts[3,9]
   - A[1,4] * Conv_1_Relu_ts[4,9] - A[1,5] * Conv_1_Relu_ts[5,9]
   - A[1,6] * Conv_1_Relu_ts[6,9] - A[1,7] * Conv_1_Relu_ts[7,9]
   + A[2,1] * Conv_2_aggregated_features[1,9]
   + A[3,1] * Conv_2_aggregated_features[1,9]
   + A[4,1] * Conv_2_aggregated_features[1,9]
   + A[5,1] * Conv_2_aggregated_features[1,9]
   + A[6,1] * Conv_2_aggregated_features[1,9]
   + A[7,1] * Conv_2_aggregated_features[1,9] ] = 0
 Conv_2_averages_constraint[1,10]: [
   A[0,1] * Conv_2_aggregated_features[1,10]
   - A[1,0] * Conv_1_Relu_ts[0,10] - A[1,1] * Conv_1_Relu_ts[1,10]
   + A[1,1] * Conv_2_aggregated_features[1,10]
   - A[1,2] * Conv_1_Relu_ts[2,10] - A[1,3] * Conv_1_Relu_ts[3,10]
   - A[1,4] * Conv_1_Relu_ts[4,10] - A[1,5] * Conv_1_Relu_ts[5,10]
   - A[1,6] * Conv_1_Relu_ts[6,10] - A[1,7] * Conv_1_Relu_ts[7,10]
   + A[2,1] * Conv_2_aggregated_features[1,10]
   + A[3,1] * Conv_2_aggregated_features[1,10]
   + A[4,1] * Conv_2_aggregated_features[1,10]
   + A[5,1] * Conv_2_aggregated_features[1,10]
   + A[6,1] * Conv_2_aggregated_features[1,10]
   + A[7,1] * Conv_2_aggregated_features[1,10] ] = 0
 Conv_2_averages_constraint[1,11]: [
   A[0,1] * Conv_2_aggregated_features[1,11]
   - A[1,0] * Conv_1_Relu_ts[0,11] - A[1,1] * Conv_1_Relu_ts[1,11]
   + A[1,1] * Conv_2_aggregated_features[1,11]
   - A[1,2] * Conv_1_Relu_ts[2,11] - A[1,3] * Conv_1_Relu_ts[3,11]
   - A[1,4] * Conv_1_Relu_ts[4,11] - A[1,5] * Conv_1_Relu_ts[5,11]
   - A[1,6] * Conv_1_Relu_ts[6,11] - A[1,7] * Conv_1_Relu_ts[7,11]
   + A[2,1] * Conv_2_aggregated_features[1,11]
   + A[3,1] * Conv_2_aggregated_features[1,11]
   + A[4,1] * Conv_2_aggregated_features[1,11]
   + A[5,1] * Conv_2_aggregated_features[1,11]
   + A[6,1] * Conv_2_aggregated_features[1,11]
   + A[7,1] * Conv_2_aggregated_features[1,11] ] = 0
 Conv_2_averages_constraint[1,12]: [
   A[0,1] * Conv_2_aggregated_features[1,12]
   - A[1,0] * Conv_1_Relu_ts[0,12] - A[1,1] * Conv_1_Relu_ts[1,12]
   + A[1,1] * Conv_2_aggregated_features[1,12]
   - A[1,2] * Conv_1_Relu_ts[2,12] - A[1,3] * Conv_1_Relu_ts[3,12]
   - A[1,4] * Conv_1_Relu_ts[4,12] - A[1,5] * Conv_1_Relu_ts[5,12]
   - A[1,6] * Conv_1_Relu_ts[6,12] - A[1,7] * Conv_1_Relu_ts[7,12]
   + A[2,1] * Conv_2_aggregated_features[1,12]
   + A[3,1] * Conv_2_aggregated_features[1,12]
   + A[4,1] * Conv_2_aggregated_features[1,12]
   + A[5,1] * Conv_2_aggregated_features[1,12]
   + A[6,1] * Conv_2_aggregated_features[1,12]
   + A[7,1] * Conv_2_aggregated_features[1,12] ] = 0
 Conv_2_averages_constraint[1,13]: [
   A[0,1] * Conv_2_aggregated_features[1,13]
   - A[1,0] * Conv_1_Relu_ts[0,13] - A[1,1] * Conv_1_Relu_ts[1,13]
   + A[1,1] * Conv_2_aggregated_features[1,13]
   - A[1,2] * Conv_1_Relu_ts[2,13] - A[1,3] * Conv_1_Relu_ts[3,13]
   - A[1,4] * Conv_1_Relu_ts[4,13] - A[1,5] * Conv_1_Relu_ts[5,13]
   - A[1,6] * Conv_1_Relu_ts[6,13] - A[1,7] * Conv_1_Relu_ts[7,13]
   + A[2,1] * Conv_2_aggregated_features[1,13]
   + A[3,1] * Conv_2_aggregated_features[1,13]
   + A[4,1] * Conv_2_aggregated_features[1,13]
   + A[5,1] * Conv_2_aggregated_features[1,13]
   + A[6,1] * Conv_2_aggregated_features[1,13]
   + A[7,1] * Conv_2_aggregated_features[1,13] ] = 0
 Conv_2_averages_constraint[1,14]: [
   A[0,1] * Conv_2_aggregated_features[1,14]
   - A[1,0] * Conv_1_Relu_ts[0,14] - A[1,1] * Conv_1_Relu_ts[1,14]
   + A[1,1] * Conv_2_aggregated_features[1,14]
   - A[1,2] * Conv_1_Relu_ts[2,14] - A[1,3] * Conv_1_Relu_ts[3,14]
   - A[1,4] * Conv_1_Relu_ts[4,14] - A[1,5] * Conv_1_Relu_ts[5,14]
   - A[1,6] * Conv_1_Relu_ts[6,14] - A[1,7] * Conv_1_Relu_ts[7,14]
   + A[2,1] * Conv_2_aggregated_features[1,14]
   + A[3,1] * Conv_2_aggregated_features[1,14]
   + A[4,1] * Conv_2_aggregated_features[1,14]
   + A[5,1] * Conv_2_aggregated_features[1,14]
   + A[6,1] * Conv_2_aggregated_features[1,14]
   + A[7,1] * Conv_2_aggregated_features[1,14] ] = 0
 Conv_2_averages_constraint[1,15]: [
   A[0,1] * Conv_2_aggregated_features[1,15]
   - A[1,0] * Conv_1_Relu_ts[0,15] - A[1,1] * Conv_1_Relu_ts[1,15]
   + A[1,1] * Conv_2_aggregated_features[1,15]
   - A[1,2] * Conv_1_Relu_ts[2,15] - A[1,3] * Conv_1_Relu_ts[3,15]
   - A[1,4] * Conv_1_Relu_ts[4,15] - A[1,5] * Conv_1_Relu_ts[5,15]
   - A[1,6] * Conv_1_Relu_ts[6,15] - A[1,7] * Conv_1_Relu_ts[7,15]
   + A[2,1] * Conv_2_aggregated_features[1,15]
   + A[3,1] * Conv_2_aggregated_features[1,15]
   + A[4,1] * Conv_2_aggregated_features[1,15]
   + A[5,1] * Conv_2_aggregated_features[1,15]
   + A[6,1] * Conv_2_aggregated_features[1,15]
   + A[7,1] * Conv_2_aggregated_features[1,15] ] = 0
 Conv_2_averages_constraint[2,0]: [
   A[0,2] * Conv_2_aggregated_features[2,0]
   + A[1,2] * Conv_2_aggregated_features[2,0]
   - A[2,0] * Conv_1_Relu_ts[0,0] - A[2,1] * Conv_1_Relu_ts[1,0]
   - A[2,2] * Conv_1_Relu_ts[2,0]
   + A[2,2] * Conv_2_aggregated_features[2,0]
   - A[2,3] * Conv_1_Relu_ts[3,0] - A[2,4] * Conv_1_Relu_ts[4,0]
   - A[2,5] * Conv_1_Relu_ts[5,0] - A[2,6] * Conv_1_Relu_ts[6,0]
   - A[2,7] * Conv_1_Relu_ts[7,0]
   + A[3,2] * Conv_2_aggregated_features[2,0]
   + A[4,2] * Conv_2_aggregated_features[2,0]
   + A[5,2] * Conv_2_aggregated_features[2,0]
   + A[6,2] * Conv_2_aggregated_features[2,0]
   + A[7,2] * Conv_2_aggregated_features[2,0] ] = 0
 Conv_2_averages_constraint[2,1]: [
   A[0,2] * Conv_2_aggregated_features[2,1]
   + A[1,2] * Conv_2_aggregated_features[2,1]
   - A[2,0] * Conv_1_Relu_ts[0,1] - A[2,1] * Conv_1_Relu_ts[1,1]
   - A[2,2] * Conv_1_Relu_ts[2,1]
   + A[2,2] * Conv_2_aggregated_features[2,1]
   - A[2,3] * Conv_1_Relu_ts[3,1] - A[2,4] * Conv_1_Relu_ts[4,1]
   - A[2,5] * Conv_1_Relu_ts[5,1] - A[2,6] * Conv_1_Relu_ts[6,1]
   - A[2,7] * Conv_1_Relu_ts[7,1]
   + A[3,2] * Conv_2_aggregated_features[2,1]
   + A[4,2] * Conv_2_aggregated_features[2,1]
   + A[5,2] * Conv_2_aggregated_features[2,1]
   + A[6,2] * Conv_2_aggregated_features[2,1]
   + A[7,2] * Conv_2_aggregated_features[2,1] ] = 0
 Conv_2_averages_constraint[2,2]: [
   A[0,2] * Conv_2_aggregated_features[2,2]
   + A[1,2] * Conv_2_aggregated_features[2,2]
   - A[2,0] * Conv_1_Relu_ts[0,2] - A[2,1] * Conv_1_Relu_ts[1,2]
   - A[2,2] * Conv_1_Relu_ts[2,2]
   + A[2,2] * Conv_2_aggregated_features[2,2]
   - A[2,3] * Conv_1_Relu_ts[3,2] - A[2,4] * Conv_1_Relu_ts[4,2]
   - A[2,5] * Conv_1_Relu_ts[5,2] - A[2,6] * Conv_1_Relu_ts[6,2]
   - A[2,7] * Conv_1_Relu_ts[7,2]
   + A[3,2] * Conv_2_aggregated_features[2,2]
   + A[4,2] * Conv_2_aggregated_features[2,2]
   + A[5,2] * Conv_2_aggregated_features[2,2]
   + A[6,2] * Conv_2_aggregated_features[2,2]
   + A[7,2] * Conv_2_aggregated_features[2,2] ] = 0
 Conv_2_averages_constraint[2,3]: [
   A[0,2] * Conv_2_aggregated_features[2,3]
   + A[1,2] * Conv_2_aggregated_features[2,3]
   - A[2,0] * Conv_1_Relu_ts[0,3] - A[2,1] * Conv_1_Relu_ts[1,3]
   - A[2,2] * Conv_1_Relu_ts[2,3]
   + A[2,2] * Conv_2_aggregated_features[2,3]
   - A[2,3] * Conv_1_Relu_ts[3,3] - A[2,4] * Conv_1_Relu_ts[4,3]
   - A[2,5] * Conv_1_Relu_ts[5,3] - A[2,6] * Conv_1_Relu_ts[6,3]
   - A[2,7] * Conv_1_Relu_ts[7,3]
   + A[3,2] * Conv_2_aggregated_features[2,3]
   + A[4,2] * Conv_2_aggregated_features[2,3]
   + A[5,2] * Conv_2_aggregated_features[2,3]
   + A[6,2] * Conv_2_aggregated_features[2,3]
   + A[7,2] * Conv_2_aggregated_features[2,3] ] = 0
 Conv_2_averages_constraint[2,4]: [
   A[0,2] * Conv_2_aggregated_features[2,4]
   + A[1,2] * Conv_2_aggregated_features[2,4]
   - A[2,0] * Conv_1_Relu_ts[0,4] - A[2,1] * Conv_1_Relu_ts[1,4]
   - A[2,2] * Conv_1_Relu_ts[2,4]
   + A[2,2] * Conv_2_aggregated_features[2,4]
   - A[2,3] * Conv_1_Relu_ts[3,4] - A[2,4] * Conv_1_Relu_ts[4,4]
   - A[2,5] * Conv_1_Relu_ts[5,4] - A[2,6] * Conv_1_Relu_ts[6,4]
   - A[2,7] * Conv_1_Relu_ts[7,4]
   + A[3,2] * Conv_2_aggregated_features[2,4]
   + A[4,2] * Conv_2_aggregated_features[2,4]
   + A[5,2] * Conv_2_aggregated_features[2,4]
   + A[6,2] * Conv_2_aggregated_features[2,4]
   + A[7,2] * Conv_2_aggregated_features[2,4] ] = 0
 Conv_2_averages_constraint[2,5]: [
   A[0,2] * Conv_2_aggregated_features[2,5]
   + A[1,2] * Conv_2_aggregated_features[2,5]
   - A[2,0] * Conv_1_Relu_ts[0,5] - A[2,1] * Conv_1_Relu_ts[1,5]
   - A[2,2] * Conv_1_Relu_ts[2,5]
   + A[2,2] * Conv_2_aggregated_features[2,5]
   - A[2,3] * Conv_1_Relu_ts[3,5] - A[2,4] * Conv_1_Relu_ts[4,5]
   - A[2,5] * Conv_1_Relu_ts[5,5] - A[2,6] * Conv_1_Relu_ts[6,5]
   - A[2,7] * Conv_1_Relu_ts[7,5]
   + A[3,2] * Conv_2_aggregated_features[2,5]
   + A[4,2] * Conv_2_aggregated_features[2,5]
   + A[5,2] * Conv_2_aggregated_features[2,5]
   + A[6,2] * Conv_2_aggregated_features[2,5]
   + A[7,2] * Conv_2_aggregated_features[2,5] ] = 0
 Conv_2_averages_constraint[2,6]: [
   A[0,2] * Conv_2_aggregated_features[2,6]
   + A[1,2] * Conv_2_aggregated_features[2,6]
   - A[2,0] * Conv_1_Relu_ts[0,6] - A[2,1] * Conv_1_Relu_ts[1,6]
   - A[2,2] * Conv_1_Relu_ts[2,6]
   + A[2,2] * Conv_2_aggregated_features[2,6]
   - A[2,3] * Conv_1_Relu_ts[3,6] - A[2,4] * Conv_1_Relu_ts[4,6]
   - A[2,5] * Conv_1_Relu_ts[5,6] - A[2,6] * Conv_1_Relu_ts[6,6]
   - A[2,7] * Conv_1_Relu_ts[7,6]
   + A[3,2] * Conv_2_aggregated_features[2,6]
   + A[4,2] * Conv_2_aggregated_features[2,6]
   + A[5,2] * Conv_2_aggregated_features[2,6]
   + A[6,2] * Conv_2_aggregated_features[2,6]
   + A[7,2] * Conv_2_aggregated_features[2,6] ] = 0
 Conv_2_averages_constraint[2,7]: [
   A[0,2] * Conv_2_aggregated_features[2,7]
   + A[1,2] * Conv_2_aggregated_features[2,7]
   - A[2,0] * Conv_1_Relu_ts[0,7] - A[2,1] * Conv_1_Relu_ts[1,7]
   - A[2,2] * Conv_1_Relu_ts[2,7]
   + A[2,2] * Conv_2_aggregated_features[2,7]
   - A[2,3] * Conv_1_Relu_ts[3,7] - A[2,4] * Conv_1_Relu_ts[4,7]
   - A[2,5] * Conv_1_Relu_ts[5,7] - A[2,6] * Conv_1_Relu_ts[6,7]
   - A[2,7] * Conv_1_Relu_ts[7,7]
   + A[3,2] * Conv_2_aggregated_features[2,7]
   + A[4,2] * Conv_2_aggregated_features[2,7]
   + A[5,2] * Conv_2_aggregated_features[2,7]
   + A[6,2] * Conv_2_aggregated_features[2,7]
   + A[7,2] * Conv_2_aggregated_features[2,7] ] = 0
 Conv_2_averages_constraint[2,8]: [
   A[0,2] * Conv_2_aggregated_features[2,8]
   + A[1,2] * Conv_2_aggregated_features[2,8]
   - A[2,0] * Conv_1_Relu_ts[0,8] - A[2,1] * Conv_1_Relu_ts[1,8]
   - A[2,2] * Conv_1_Relu_ts[2,8]
   + A[2,2] * Conv_2_aggregated_features[2,8]
   - A[2,3] * Conv_1_Relu_ts[3,8] - A[2,4] * Conv_1_Relu_ts[4,8]
   - A[2,5] * Conv_1_Relu_ts[5,8] - A[2,6] * Conv_1_Relu_ts[6,8]
   - A[2,7] * Conv_1_Relu_ts[7,8]
   + A[3,2] * Conv_2_aggregated_features[2,8]
   + A[4,2] * Conv_2_aggregated_features[2,8]
   + A[5,2] * Conv_2_aggregated_features[2,8]
   + A[6,2] * Conv_2_aggregated_features[2,8]
   + A[7,2] * Conv_2_aggregated_features[2,8] ] = 0
 Conv_2_averages_constraint[2,9]: [
   A[0,2] * Conv_2_aggregated_features[2,9]
   + A[1,2] * Conv_2_aggregated_features[2,9]
   - A[2,0] * Conv_1_Relu_ts[0,9] - A[2,1] * Conv_1_Relu_ts[1,9]
   - A[2,2] * Conv_1_Relu_ts[2,9]
   + A[2,2] * Conv_2_aggregated_features[2,9]
   - A[2,3] * Conv_1_Relu_ts[3,9] - A[2,4] * Conv_1_Relu_ts[4,9]
   - A[2,5] * Conv_1_Relu_ts[5,9] - A[2,6] * Conv_1_Relu_ts[6,9]
   - A[2,7] * Conv_1_Relu_ts[7,9]
   + A[3,2] * Conv_2_aggregated_features[2,9]
   + A[4,2] * Conv_2_aggregated_features[2,9]
   + A[5,2] * Conv_2_aggregated_features[2,9]
   + A[6,2] * Conv_2_aggregated_features[2,9]
   + A[7,2] * Conv_2_aggregated_features[2,9] ] = 0
 Conv_2_averages_constraint[2,10]: [
   A[0,2] * Conv_2_aggregated_features[2,10]
   + A[1,2] * Conv_2_aggregated_features[2,10]
   - A[2,0] * Conv_1_Relu_ts[0,10] - A[2,1] * Conv_1_Relu_ts[1,10]
   - A[2,2] * Conv_1_Relu_ts[2,10]
   + A[2,2] * Conv_2_aggregated_features[2,10]
   - A[2,3] * Conv_1_Relu_ts[3,10] - A[2,4] * Conv_1_Relu_ts[4,10]
   - A[2,5] * Conv_1_Relu_ts[5,10] - A[2,6] * Conv_1_Relu_ts[6,10]
   - A[2,7] * Conv_1_Relu_ts[7,10]
   + A[3,2] * Conv_2_aggregated_features[2,10]
   + A[4,2] * Conv_2_aggregated_features[2,10]
   + A[5,2] * Conv_2_aggregated_features[2,10]
   + A[6,2] * Conv_2_aggregated_features[2,10]
   + A[7,2] * Conv_2_aggregated_features[2,10] ] = 0
 Conv_2_averages_constraint[2,11]: [
   A[0,2] * Conv_2_aggregated_features[2,11]
   + A[1,2] * Conv_2_aggregated_features[2,11]
   - A[2,0] * Conv_1_Relu_ts[0,11] - A[2,1] * Conv_1_Relu_ts[1,11]
   - A[2,2] * Conv_1_Relu_ts[2,11]
   + A[2,2] * Conv_2_aggregated_features[2,11]
   - A[2,3] * Conv_1_Relu_ts[3,11] - A[2,4] * Conv_1_Relu_ts[4,11]
   - A[2,5] * Conv_1_Relu_ts[5,11] - A[2,6] * Conv_1_Relu_ts[6,11]
   - A[2,7] * Conv_1_Relu_ts[7,11]
   + A[3,2] * Conv_2_aggregated_features[2,11]
   + A[4,2] * Conv_2_aggregated_features[2,11]
   + A[5,2] * Conv_2_aggregated_features[2,11]
   + A[6,2] * Conv_2_aggregated_features[2,11]
   + A[7,2] * Conv_2_aggregated_features[2,11] ] = 0
 Conv_2_averages_constraint[2,12]: [
   A[0,2] * Conv_2_aggregated_features[2,12]
   + A[1,2] * Conv_2_aggregated_features[2,12]
   - A[2,0] * Conv_1_Relu_ts[0,12] - A[2,1] * Conv_1_Relu_ts[1,12]
   - A[2,2] * Conv_1_Relu_ts[2,12]
   + A[2,2] * Conv_2_aggregated_features[2,12]
   - A[2,3] * Conv_1_Relu_ts[3,12] - A[2,4] * Conv_1_Relu_ts[4,12]
   - A[2,5] * Conv_1_Relu_ts[5,12] - A[2,6] * Conv_1_Relu_ts[6,12]
   - A[2,7] * Conv_1_Relu_ts[7,12]
   + A[3,2] * Conv_2_aggregated_features[2,12]
   + A[4,2] * Conv_2_aggregated_features[2,12]
   + A[5,2] * Conv_2_aggregated_features[2,12]
   + A[6,2] * Conv_2_aggregated_features[2,12]
   + A[7,2] * Conv_2_aggregated_features[2,12] ] = 0
 Conv_2_averages_constraint[2,13]: [
   A[0,2] * Conv_2_aggregated_features[2,13]
   + A[1,2] * Conv_2_aggregated_features[2,13]
   - A[2,0] * Conv_1_Relu_ts[0,13] - A[2,1] * Conv_1_Relu_ts[1,13]
   - A[2,2] * Conv_1_Relu_ts[2,13]
   + A[2,2] * Conv_2_aggregated_features[2,13]
   - A[2,3] * Conv_1_Relu_ts[3,13] - A[2,4] * Conv_1_Relu_ts[4,13]
   - A[2,5] * Conv_1_Relu_ts[5,13] - A[2,6] * Conv_1_Relu_ts[6,13]
   - A[2,7] * Conv_1_Relu_ts[7,13]
   + A[3,2] * Conv_2_aggregated_features[2,13]
   + A[4,2] * Conv_2_aggregated_features[2,13]
   + A[5,2] * Conv_2_aggregated_features[2,13]
   + A[6,2] * Conv_2_aggregated_features[2,13]
   + A[7,2] * Conv_2_aggregated_features[2,13] ] = 0
 Conv_2_averages_constraint[2,14]: [
   A[0,2] * Conv_2_aggregated_features[2,14]
   + A[1,2] * Conv_2_aggregated_features[2,14]
   - A[2,0] * Conv_1_Relu_ts[0,14] - A[2,1] * Conv_1_Relu_ts[1,14]
   - A[2,2] * Conv_1_Relu_ts[2,14]
   + A[2,2] * Conv_2_aggregated_features[2,14]
   - A[2,3] * Conv_1_Relu_ts[3,14] - A[2,4] * Conv_1_Relu_ts[4,14]
   - A[2,5] * Conv_1_Relu_ts[5,14] - A[2,6] * Conv_1_Relu_ts[6,14]
   - A[2,7] * Conv_1_Relu_ts[7,14]
   + A[3,2] * Conv_2_aggregated_features[2,14]
   + A[4,2] * Conv_2_aggregated_features[2,14]
   + A[5,2] * Conv_2_aggregated_features[2,14]
   + A[6,2] * Conv_2_aggregated_features[2,14]
   + A[7,2] * Conv_2_aggregated_features[2,14] ] = 0
 Conv_2_averages_constraint[2,15]: [
   A[0,2] * Conv_2_aggregated_features[2,15]
   + A[1,2] * Conv_2_aggregated_features[2,15]
   - A[2,0] * Conv_1_Relu_ts[0,15] - A[2,1] * Conv_1_Relu_ts[1,15]
   - A[2,2] * Conv_1_Relu_ts[2,15]
   + A[2,2] * Conv_2_aggregated_features[2,15]
   - A[2,3] * Conv_1_Relu_ts[3,15] - A[2,4] * Conv_1_Relu_ts[4,15]
   - A[2,5] * Conv_1_Relu_ts[5,15] - A[2,6] * Conv_1_Relu_ts[6,15]
   - A[2,7] * Conv_1_Relu_ts[7,15]
   + A[3,2] * Conv_2_aggregated_features[2,15]
   + A[4,2] * Conv_2_aggregated_features[2,15]
   + A[5,2] * Conv_2_aggregated_features[2,15]
   + A[6,2] * Conv_2_aggregated_features[2,15]
   + A[7,2] * Conv_2_aggregated_features[2,15] ] = 0
 Conv_2_averages_constraint[3,0]: [
   A[0,3] * Conv_2_aggregated_features[3,0]
   + A[1,3] * Conv_2_aggregated_features[3,0]
   + A[2,3] * Conv_2_aggregated_features[3,0]
   - A[3,0] * Conv_1_Relu_ts[0,0] - A[3,1] * Conv_1_Relu_ts[1,0]
   - A[3,2] * Conv_1_Relu_ts[2,0] - A[3,3] * Conv_1_Relu_ts[3,0]
   + A[3,3] * Conv_2_aggregated_features[3,0]
   - A[3,4] * Conv_1_Relu_ts[4,0] - A[3,5] * Conv_1_Relu_ts[5,0]
   - A[3,6] * Conv_1_Relu_ts[6,0] - A[3,7] * Conv_1_Relu_ts[7,0]
   + A[4,3] * Conv_2_aggregated_features[3,0]
   + A[5,3] * Conv_2_aggregated_features[3,0]
   + A[6,3] * Conv_2_aggregated_features[3,0]
   + A[7,3] * Conv_2_aggregated_features[3,0] ] = 0
 Conv_2_averages_constraint[3,1]: [
   A[0,3] * Conv_2_aggregated_features[3,1]
   + A[1,3] * Conv_2_aggregated_features[3,1]
   + A[2,3] * Conv_2_aggregated_features[3,1]
   - A[3,0] * Conv_1_Relu_ts[0,1] - A[3,1] * Conv_1_Relu_ts[1,1]
   - A[3,2] * Conv_1_Relu_ts[2,1] - A[3,3] * Conv_1_Relu_ts[3,1]
   + A[3,3] * Conv_2_aggregated_features[3,1]
   - A[3,4] * Conv_1_Relu_ts[4,1] - A[3,5] * Conv_1_Relu_ts[5,1]
   - A[3,6] * Conv_1_Relu_ts[6,1] - A[3,7] * Conv_1_Relu_ts[7,1]
   + A[4,3] * Conv_2_aggregated_features[3,1]
   + A[5,3] * Conv_2_aggregated_features[3,1]
   + A[6,3] * Conv_2_aggregated_features[3,1]
   + A[7,3] * Conv_2_aggregated_features[3,1] ] = 0
 Conv_2_averages_constraint[3,2]: [
   A[0,3] * Conv_2_aggregated_features[3,2]
   + A[1,3] * Conv_2_aggregated_features[3,2]
   + A[2,3] * Conv_2_aggregated_features[3,2]
   - A[3,0] * Conv_1_Relu_ts[0,2] - A[3,1] * Conv_1_Relu_ts[1,2]
   - A[3,2] * Conv_1_Relu_ts[2,2] - A[3,3] * Conv_1_Relu_ts[3,2]
   + A[3,3] * Conv_2_aggregated_features[3,2]
   - A[3,4] * Conv_1_Relu_ts[4,2] - A[3,5] * Conv_1_Relu_ts[5,2]
   - A[3,6] * Conv_1_Relu_ts[6,2] - A[3,7] * Conv_1_Relu_ts[7,2]
   + A[4,3] * Conv_2_aggregated_features[3,2]
   + A[5,3] * Conv_2_aggregated_features[3,2]
   + A[6,3] * Conv_2_aggregated_features[3,2]
   + A[7,3] * Conv_2_aggregated_features[3,2] ] = 0
 Conv_2_averages_constraint[3,3]: [
   A[0,3] * Conv_2_aggregated_features[3,3]
   + A[1,3] * Conv_2_aggregated_features[3,3]
   + A[2,3] * Conv_2_aggregated_features[3,3]
   - A[3,0] * Conv_1_Relu_ts[0,3] - A[3,1] * Conv_1_Relu_ts[1,3]
   - A[3,2] * Conv_1_Relu_ts[2,3] - A[3,3] * Conv_1_Relu_ts[3,3]
   + A[3,3] * Conv_2_aggregated_features[3,3]
   - A[3,4] * Conv_1_Relu_ts[4,3] - A[3,5] * Conv_1_Relu_ts[5,3]
   - A[3,6] * Conv_1_Relu_ts[6,3] - A[3,7] * Conv_1_Relu_ts[7,3]
   + A[4,3] * Conv_2_aggregated_features[3,3]
   + A[5,3] * Conv_2_aggregated_features[3,3]
   + A[6,3] * Conv_2_aggregated_features[3,3]
   + A[7,3] * Conv_2_aggregated_features[3,3] ] = 0
 Conv_2_averages_constraint[3,4]: [
   A[0,3] * Conv_2_aggregated_features[3,4]
   + A[1,3] * Conv_2_aggregated_features[3,4]
   + A[2,3] * Conv_2_aggregated_features[3,4]
   - A[3,0] * Conv_1_Relu_ts[0,4] - A[3,1] * Conv_1_Relu_ts[1,4]
   - A[3,2] * Conv_1_Relu_ts[2,4] - A[3,3] * Conv_1_Relu_ts[3,4]
   + A[3,3] * Conv_2_aggregated_features[3,4]
   - A[3,4] * Conv_1_Relu_ts[4,4] - A[3,5] * Conv_1_Relu_ts[5,4]
   - A[3,6] * Conv_1_Relu_ts[6,4] - A[3,7] * Conv_1_Relu_ts[7,4]
   + A[4,3] * Conv_2_aggregated_features[3,4]
   + A[5,3] * Conv_2_aggregated_features[3,4]
   + A[6,3] * Conv_2_aggregated_features[3,4]
   + A[7,3] * Conv_2_aggregated_features[3,4] ] = 0
 Conv_2_averages_constraint[3,5]: [
   A[0,3] * Conv_2_aggregated_features[3,5]
   + A[1,3] * Conv_2_aggregated_features[3,5]
   + A[2,3] * Conv_2_aggregated_features[3,5]
   - A[3,0] * Conv_1_Relu_ts[0,5] - A[3,1] * Conv_1_Relu_ts[1,5]
   - A[3,2] * Conv_1_Relu_ts[2,5] - A[3,3] * Conv_1_Relu_ts[3,5]
   + A[3,3] * Conv_2_aggregated_features[3,5]
   - A[3,4] * Conv_1_Relu_ts[4,5] - A[3,5] * Conv_1_Relu_ts[5,5]
   - A[3,6] * Conv_1_Relu_ts[6,5] - A[3,7] * Conv_1_Relu_ts[7,5]
   + A[4,3] * Conv_2_aggregated_features[3,5]
   + A[5,3] * Conv_2_aggregated_features[3,5]
   + A[6,3] * Conv_2_aggregated_features[3,5]
   + A[7,3] * Conv_2_aggregated_features[3,5] ] = 0
 Conv_2_averages_constraint[3,6]: [
   A[0,3] * Conv_2_aggregated_features[3,6]
   + A[1,3] * Conv_2_aggregated_features[3,6]
   + A[2,3] * Conv_2_aggregated_features[3,6]
   - A[3,0] * Conv_1_Relu_ts[0,6] - A[3,1] * Conv_1_Relu_ts[1,6]
   - A[3,2] * Conv_1_Relu_ts[2,6] - A[3,3] * Conv_1_Relu_ts[3,6]
   + A[3,3] * Conv_2_aggregated_features[3,6]
   - A[3,4] * Conv_1_Relu_ts[4,6] - A[3,5] * Conv_1_Relu_ts[5,6]
   - A[3,6] * Conv_1_Relu_ts[6,6] - A[3,7] * Conv_1_Relu_ts[7,6]
   + A[4,3] * Conv_2_aggregated_features[3,6]
   + A[5,3] * Conv_2_aggregated_features[3,6]
   + A[6,3] * Conv_2_aggregated_features[3,6]
   + A[7,3] * Conv_2_aggregated_features[3,6] ] = 0
 Conv_2_averages_constraint[3,7]: [
   A[0,3] * Conv_2_aggregated_features[3,7]
   + A[1,3] * Conv_2_aggregated_features[3,7]
   + A[2,3] * Conv_2_aggregated_features[3,7]
   - A[3,0] * Conv_1_Relu_ts[0,7] - A[3,1] * Conv_1_Relu_ts[1,7]
   - A[3,2] * Conv_1_Relu_ts[2,7] - A[3,3] * Conv_1_Relu_ts[3,7]
   + A[3,3] * Conv_2_aggregated_features[3,7]
   - A[3,4] * Conv_1_Relu_ts[4,7] - A[3,5] * Conv_1_Relu_ts[5,7]
   - A[3,6] * Conv_1_Relu_ts[6,7] - A[3,7] * Conv_1_Relu_ts[7,7]
   + A[4,3] * Conv_2_aggregated_features[3,7]
   + A[5,3] * Conv_2_aggregated_features[3,7]
   + A[6,3] * Conv_2_aggregated_features[3,7]
   + A[7,3] * Conv_2_aggregated_features[3,7] ] = 0
 Conv_2_averages_constraint[3,8]: [
   A[0,3] * Conv_2_aggregated_features[3,8]
   + A[1,3] * Conv_2_aggregated_features[3,8]
   + A[2,3] * Conv_2_aggregated_features[3,8]
   - A[3,0] * Conv_1_Relu_ts[0,8] - A[3,1] * Conv_1_Relu_ts[1,8]
   - A[3,2] * Conv_1_Relu_ts[2,8] - A[3,3] * Conv_1_Relu_ts[3,8]
   + A[3,3] * Conv_2_aggregated_features[3,8]
   - A[3,4] * Conv_1_Relu_ts[4,8] - A[3,5] * Conv_1_Relu_ts[5,8]
   - A[3,6] * Conv_1_Relu_ts[6,8] - A[3,7] * Conv_1_Relu_ts[7,8]
   + A[4,3] * Conv_2_aggregated_features[3,8]
   + A[5,3] * Conv_2_aggregated_features[3,8]
   + A[6,3] * Conv_2_aggregated_features[3,8]
   + A[7,3] * Conv_2_aggregated_features[3,8] ] = 0
 Conv_2_averages_constraint[3,9]: [
   A[0,3] * Conv_2_aggregated_features[3,9]
   + A[1,3] * Conv_2_aggregated_features[3,9]
   + A[2,3] * Conv_2_aggregated_features[3,9]
   - A[3,0] * Conv_1_Relu_ts[0,9] - A[3,1] * Conv_1_Relu_ts[1,9]
   - A[3,2] * Conv_1_Relu_ts[2,9] - A[3,3] * Conv_1_Relu_ts[3,9]
   + A[3,3] * Conv_2_aggregated_features[3,9]
   - A[3,4] * Conv_1_Relu_ts[4,9] - A[3,5] * Conv_1_Relu_ts[5,9]
   - A[3,6] * Conv_1_Relu_ts[6,9] - A[3,7] * Conv_1_Relu_ts[7,9]
   + A[4,3] * Conv_2_aggregated_features[3,9]
   + A[5,3] * Conv_2_aggregated_features[3,9]
   + A[6,3] * Conv_2_aggregated_features[3,9]
   + A[7,3] * Conv_2_aggregated_features[3,9] ] = 0
 Conv_2_averages_constraint[3,10]: [
   A[0,3] * Conv_2_aggregated_features[3,10]
   + A[1,3] * Conv_2_aggregated_features[3,10]
   + A[2,3] * Conv_2_aggregated_features[3,10]
   - A[3,0] * Conv_1_Relu_ts[0,10] - A[3,1] * Conv_1_Relu_ts[1,10]
   - A[3,2] * Conv_1_Relu_ts[2,10] - A[3,3] * Conv_1_Relu_ts[3,10]
   + A[3,3] * Conv_2_aggregated_features[3,10]
   - A[3,4] * Conv_1_Relu_ts[4,10] - A[3,5] * Conv_1_Relu_ts[5,10]
   - A[3,6] * Conv_1_Relu_ts[6,10] - A[3,7] * Conv_1_Relu_ts[7,10]
   + A[4,3] * Conv_2_aggregated_features[3,10]
   + A[5,3] * Conv_2_aggregated_features[3,10]
   + A[6,3] * Conv_2_aggregated_features[3,10]
   + A[7,3] * Conv_2_aggregated_features[3,10] ] = 0
 Conv_2_averages_constraint[3,11]: [
   A[0,3] * Conv_2_aggregated_features[3,11]
   + A[1,3] * Conv_2_aggregated_features[3,11]
   + A[2,3] * Conv_2_aggregated_features[3,11]
   - A[3,0] * Conv_1_Relu_ts[0,11] - A[3,1] * Conv_1_Relu_ts[1,11]
   - A[3,2] * Conv_1_Relu_ts[2,11] - A[3,3] * Conv_1_Relu_ts[3,11]
   + A[3,3] * Conv_2_aggregated_features[3,11]
   - A[3,4] * Conv_1_Relu_ts[4,11] - A[3,5] * Conv_1_Relu_ts[5,11]
   - A[3,6] * Conv_1_Relu_ts[6,11] - A[3,7] * Conv_1_Relu_ts[7,11]
   + A[4,3] * Conv_2_aggregated_features[3,11]
   + A[5,3] * Conv_2_aggregated_features[3,11]
   + A[6,3] * Conv_2_aggregated_features[3,11]
   + A[7,3] * Conv_2_aggregated_features[3,11] ] = 0
 Conv_2_averages_constraint[3,12]: [
   A[0,3] * Conv_2_aggregated_features[3,12]
   + A[1,3] * Conv_2_aggregated_features[3,12]
   + A[2,3] * Conv_2_aggregated_features[3,12]
   - A[3,0] * Conv_1_Relu_ts[0,12] - A[3,1] * Conv_1_Relu_ts[1,12]
   - A[3,2] * Conv_1_Relu_ts[2,12] - A[3,3] * Conv_1_Relu_ts[3,12]
   + A[3,3] * Conv_2_aggregated_features[3,12]
   - A[3,4] * Conv_1_Relu_ts[4,12] - A[3,5] * Conv_1_Relu_ts[5,12]
   - A[3,6] * Conv_1_Relu_ts[6,12] - A[3,7] * Conv_1_Relu_ts[7,12]
   + A[4,3] * Conv_2_aggregated_features[3,12]
   + A[5,3] * Conv_2_aggregated_features[3,12]
   + A[6,3] * Conv_2_aggregated_features[3,12]
   + A[7,3] * Conv_2_aggregated_features[3,12] ] = 0
 Conv_2_averages_constraint[3,13]: [
   A[0,3] * Conv_2_aggregated_features[3,13]
   + A[1,3] * Conv_2_aggregated_features[3,13]
   + A[2,3] * Conv_2_aggregated_features[3,13]
   - A[3,0] * Conv_1_Relu_ts[0,13] - A[3,1] * Conv_1_Relu_ts[1,13]
   - A[3,2] * Conv_1_Relu_ts[2,13] - A[3,3] * Conv_1_Relu_ts[3,13]
   + A[3,3] * Conv_2_aggregated_features[3,13]
   - A[3,4] * Conv_1_Relu_ts[4,13] - A[3,5] * Conv_1_Relu_ts[5,13]
   - A[3,6] * Conv_1_Relu_ts[6,13] - A[3,7] * Conv_1_Relu_ts[7,13]
   + A[4,3] * Conv_2_aggregated_features[3,13]
   + A[5,3] * Conv_2_aggregated_features[3,13]
   + A[6,3] * Conv_2_aggregated_features[3,13]
   + A[7,3] * Conv_2_aggregated_features[3,13] ] = 0
 Conv_2_averages_constraint[3,14]: [
   A[0,3] * Conv_2_aggregated_features[3,14]
   + A[1,3] * Conv_2_aggregated_features[3,14]
   + A[2,3] * Conv_2_aggregated_features[3,14]
   - A[3,0] * Conv_1_Relu_ts[0,14] - A[3,1] * Conv_1_Relu_ts[1,14]
   - A[3,2] * Conv_1_Relu_ts[2,14] - A[3,3] * Conv_1_Relu_ts[3,14]
   + A[3,3] * Conv_2_aggregated_features[3,14]
   - A[3,4] * Conv_1_Relu_ts[4,14] - A[3,5] * Conv_1_Relu_ts[5,14]
   - A[3,6] * Conv_1_Relu_ts[6,14] - A[3,7] * Conv_1_Relu_ts[7,14]
   + A[4,3] * Conv_2_aggregated_features[3,14]
   + A[5,3] * Conv_2_aggregated_features[3,14]
   + A[6,3] * Conv_2_aggregated_features[3,14]
   + A[7,3] * Conv_2_aggregated_features[3,14] ] = 0
 Conv_2_averages_constraint[3,15]: [
   A[0,3] * Conv_2_aggregated_features[3,15]
   + A[1,3] * Conv_2_aggregated_features[3,15]
   + A[2,3] * Conv_2_aggregated_features[3,15]
   - A[3,0] * Conv_1_Relu_ts[0,15] - A[3,1] * Conv_1_Relu_ts[1,15]
   - A[3,2] * Conv_1_Relu_ts[2,15] - A[3,3] * Conv_1_Relu_ts[3,15]
   + A[3,3] * Conv_2_aggregated_features[3,15]
   - A[3,4] * Conv_1_Relu_ts[4,15] - A[3,5] * Conv_1_Relu_ts[5,15]
   - A[3,6] * Conv_1_Relu_ts[6,15] - A[3,7] * Conv_1_Relu_ts[7,15]
   + A[4,3] * Conv_2_aggregated_features[3,15]
   + A[5,3] * Conv_2_aggregated_features[3,15]
   + A[6,3] * Conv_2_aggregated_features[3,15]
   + A[7,3] * Conv_2_aggregated_features[3,15] ] = 0
 Conv_2_averages_constraint[4,0]: [
   A[0,4] * Conv_2_aggregated_features[4,0]
   + A[1,4] * Conv_2_aggregated_features[4,0]
   + A[2,4] * Conv_2_aggregated_features[4,0]
   + A[3,4] * Conv_2_aggregated_features[4,0]
   - A[4,0] * Conv_1_Relu_ts[0,0] - A[4,1] * Conv_1_Relu_ts[1,0]
   - A[4,2] * Conv_1_Relu_ts[2,0] - A[4,3] * Conv_1_Relu_ts[3,0]
   - A[4,4] * Conv_1_Relu_ts[4,0]
   + A[4,4] * Conv_2_aggregated_features[4,0]
   - A[4,5] * Conv_1_Relu_ts[5,0] - A[4,6] * Conv_1_Relu_ts[6,0]
   - A[4,7] * Conv_1_Relu_ts[7,0]
   + A[5,4] * Conv_2_aggregated_features[4,0]
   + A[6,4] * Conv_2_aggregated_features[4,0]
   + A[7,4] * Conv_2_aggregated_features[4,0] ] = 0
 Conv_2_averages_constraint[4,1]: [
   A[0,4] * Conv_2_aggregated_features[4,1]
   + A[1,4] * Conv_2_aggregated_features[4,1]
   + A[2,4] * Conv_2_aggregated_features[4,1]
   + A[3,4] * Conv_2_aggregated_features[4,1]
   - A[4,0] * Conv_1_Relu_ts[0,1] - A[4,1] * Conv_1_Relu_ts[1,1]
   - A[4,2] * Conv_1_Relu_ts[2,1] - A[4,3] * Conv_1_Relu_ts[3,1]
   - A[4,4] * Conv_1_Relu_ts[4,1]
   + A[4,4] * Conv_2_aggregated_features[4,1]
   - A[4,5] * Conv_1_Relu_ts[5,1] - A[4,6] * Conv_1_Relu_ts[6,1]
   - A[4,7] * Conv_1_Relu_ts[7,1]
   + A[5,4] * Conv_2_aggregated_features[4,1]
   + A[6,4] * Conv_2_aggregated_features[4,1]
   + A[7,4] * Conv_2_aggregated_features[4,1] ] = 0
 Conv_2_averages_constraint[4,2]: [
   A[0,4] * Conv_2_aggregated_features[4,2]
   + A[1,4] * Conv_2_aggregated_features[4,2]
   + A[2,4] * Conv_2_aggregated_features[4,2]
   + A[3,4] * Conv_2_aggregated_features[4,2]
   - A[4,0] * Conv_1_Relu_ts[0,2] - A[4,1] * Conv_1_Relu_ts[1,2]
   - A[4,2] * Conv_1_Relu_ts[2,2] - A[4,3] * Conv_1_Relu_ts[3,2]
   - A[4,4] * Conv_1_Relu_ts[4,2]
   + A[4,4] * Conv_2_aggregated_features[4,2]
   - A[4,5] * Conv_1_Relu_ts[5,2] - A[4,6] * Conv_1_Relu_ts[6,2]
   - A[4,7] * Conv_1_Relu_ts[7,2]
   + A[5,4] * Conv_2_aggregated_features[4,2]
   + A[6,4] * Conv_2_aggregated_features[4,2]
   + A[7,4] * Conv_2_aggregated_features[4,2] ] = 0
 Conv_2_averages_constraint[4,3]: [
   A[0,4] * Conv_2_aggregated_features[4,3]
   + A[1,4] * Conv_2_aggregated_features[4,3]
   + A[2,4] * Conv_2_aggregated_features[4,3]
   + A[3,4] * Conv_2_aggregated_features[4,3]
   - A[4,0] * Conv_1_Relu_ts[0,3] - A[4,1] * Conv_1_Relu_ts[1,3]
   - A[4,2] * Conv_1_Relu_ts[2,3] - A[4,3] * Conv_1_Relu_ts[3,3]
   - A[4,4] * Conv_1_Relu_ts[4,3]
   + A[4,4] * Conv_2_aggregated_features[4,3]
   - A[4,5] * Conv_1_Relu_ts[5,3] - A[4,6] * Conv_1_Relu_ts[6,3]
   - A[4,7] * Conv_1_Relu_ts[7,3]
   + A[5,4] * Conv_2_aggregated_features[4,3]
   + A[6,4] * Conv_2_aggregated_features[4,3]
   + A[7,4] * Conv_2_aggregated_features[4,3] ] = 0
 Conv_2_averages_constraint[4,4]: [
   A[0,4] * Conv_2_aggregated_features[4,4]
   + A[1,4] * Conv_2_aggregated_features[4,4]
   + A[2,4] * Conv_2_aggregated_features[4,4]
   + A[3,4] * Conv_2_aggregated_features[4,4]
   - A[4,0] * Conv_1_Relu_ts[0,4] - A[4,1] * Conv_1_Relu_ts[1,4]
   - A[4,2] * Conv_1_Relu_ts[2,4] - A[4,3] * Conv_1_Relu_ts[3,4]
   - A[4,4] * Conv_1_Relu_ts[4,4]
   + A[4,4] * Conv_2_aggregated_features[4,4]
   - A[4,5] * Conv_1_Relu_ts[5,4] - A[4,6] * Conv_1_Relu_ts[6,4]
   - A[4,7] * Conv_1_Relu_ts[7,4]
   + A[5,4] * Conv_2_aggregated_features[4,4]
   + A[6,4] * Conv_2_aggregated_features[4,4]
   + A[7,4] * Conv_2_aggregated_features[4,4] ] = 0
 Conv_2_averages_constraint[4,5]: [
   A[0,4] * Conv_2_aggregated_features[4,5]
   + A[1,4] * Conv_2_aggregated_features[4,5]
   + A[2,4] * Conv_2_aggregated_features[4,5]
   + A[3,4] * Conv_2_aggregated_features[4,5]
   - A[4,0] * Conv_1_Relu_ts[0,5] - A[4,1] * Conv_1_Relu_ts[1,5]
   - A[4,2] * Conv_1_Relu_ts[2,5] - A[4,3] * Conv_1_Relu_ts[3,5]
   - A[4,4] * Conv_1_Relu_ts[4,5]
   + A[4,4] * Conv_2_aggregated_features[4,5]
   - A[4,5] * Conv_1_Relu_ts[5,5] - A[4,6] * Conv_1_Relu_ts[6,5]
   - A[4,7] * Conv_1_Relu_ts[7,5]
   + A[5,4] * Conv_2_aggregated_features[4,5]
   + A[6,4] * Conv_2_aggregated_features[4,5]
   + A[7,4] * Conv_2_aggregated_features[4,5] ] = 0
 Conv_2_averages_constraint[4,6]: [
   A[0,4] * Conv_2_aggregated_features[4,6]
   + A[1,4] * Conv_2_aggregated_features[4,6]
   + A[2,4] * Conv_2_aggregated_features[4,6]
   + A[3,4] * Conv_2_aggregated_features[4,6]
   - A[4,0] * Conv_1_Relu_ts[0,6] - A[4,1] * Conv_1_Relu_ts[1,6]
   - A[4,2] * Conv_1_Relu_ts[2,6] - A[4,3] * Conv_1_Relu_ts[3,6]
   - A[4,4] * Conv_1_Relu_ts[4,6]
   + A[4,4] * Conv_2_aggregated_features[4,6]
   - A[4,5] * Conv_1_Relu_ts[5,6] - A[4,6] * Conv_1_Relu_ts[6,6]
   - A[4,7] * Conv_1_Relu_ts[7,6]
   + A[5,4] * Conv_2_aggregated_features[4,6]
   + A[6,4] * Conv_2_aggregated_features[4,6]
   + A[7,4] * Conv_2_aggregated_features[4,6] ] = 0
 Conv_2_averages_constraint[4,7]: [
   A[0,4] * Conv_2_aggregated_features[4,7]
   + A[1,4] * Conv_2_aggregated_features[4,7]
   + A[2,4] * Conv_2_aggregated_features[4,7]
   + A[3,4] * Conv_2_aggregated_features[4,7]
   - A[4,0] * Conv_1_Relu_ts[0,7] - A[4,1] * Conv_1_Relu_ts[1,7]
   - A[4,2] * Conv_1_Relu_ts[2,7] - A[4,3] * Conv_1_Relu_ts[3,7]
   - A[4,4] * Conv_1_Relu_ts[4,7]
   + A[4,4] * Conv_2_aggregated_features[4,7]
   - A[4,5] * Conv_1_Relu_ts[5,7] - A[4,6] * Conv_1_Relu_ts[6,7]
   - A[4,7] * Conv_1_Relu_ts[7,7]
   + A[5,4] * Conv_2_aggregated_features[4,7]
   + A[6,4] * Conv_2_aggregated_features[4,7]
   + A[7,4] * Conv_2_aggregated_features[4,7] ] = 0
 Conv_2_averages_constraint[4,8]: [
   A[0,4] * Conv_2_aggregated_features[4,8]
   + A[1,4] * Conv_2_aggregated_features[4,8]
   + A[2,4] * Conv_2_aggregated_features[4,8]
   + A[3,4] * Conv_2_aggregated_features[4,8]
   - A[4,0] * Conv_1_Relu_ts[0,8] - A[4,1] * Conv_1_Relu_ts[1,8]
   - A[4,2] * Conv_1_Relu_ts[2,8] - A[4,3] * Conv_1_Relu_ts[3,8]
   - A[4,4] * Conv_1_Relu_ts[4,8]
   + A[4,4] * Conv_2_aggregated_features[4,8]
   - A[4,5] * Conv_1_Relu_ts[5,8] - A[4,6] * Conv_1_Relu_ts[6,8]
   - A[4,7] * Conv_1_Relu_ts[7,8]
   + A[5,4] * Conv_2_aggregated_features[4,8]
   + A[6,4] * Conv_2_aggregated_features[4,8]
   + A[7,4] * Conv_2_aggregated_features[4,8] ] = 0
 Conv_2_averages_constraint[4,9]: [
   A[0,4] * Conv_2_aggregated_features[4,9]
   + A[1,4] * Conv_2_aggregated_features[4,9]
   + A[2,4] * Conv_2_aggregated_features[4,9]
   + A[3,4] * Conv_2_aggregated_features[4,9]
   - A[4,0] * Conv_1_Relu_ts[0,9] - A[4,1] * Conv_1_Relu_ts[1,9]
   - A[4,2] * Conv_1_Relu_ts[2,9] - A[4,3] * Conv_1_Relu_ts[3,9]
   - A[4,4] * Conv_1_Relu_ts[4,9]
   + A[4,4] * Conv_2_aggregated_features[4,9]
   - A[4,5] * Conv_1_Relu_ts[5,9] - A[4,6] * Conv_1_Relu_ts[6,9]
   - A[4,7] * Conv_1_Relu_ts[7,9]
   + A[5,4] * Conv_2_aggregated_features[4,9]
   + A[6,4] * Conv_2_aggregated_features[4,9]
   + A[7,4] * Conv_2_aggregated_features[4,9] ] = 0
 Conv_2_averages_constraint[4,10]: [
   A[0,4] * Conv_2_aggregated_features[4,10]
   + A[1,4] * Conv_2_aggregated_features[4,10]
   + A[2,4] * Conv_2_aggregated_features[4,10]
   + A[3,4] * Conv_2_aggregated_features[4,10]
   - A[4,0] * Conv_1_Relu_ts[0,10] - A[4,1] * Conv_1_Relu_ts[1,10]
   - A[4,2] * Conv_1_Relu_ts[2,10] - A[4,3] * Conv_1_Relu_ts[3,10]
   - A[4,4] * Conv_1_Relu_ts[4,10]
   + A[4,4] * Conv_2_aggregated_features[4,10]
   - A[4,5] * Conv_1_Relu_ts[5,10] - A[4,6] * Conv_1_Relu_ts[6,10]
   - A[4,7] * Conv_1_Relu_ts[7,10]
   + A[5,4] * Conv_2_aggregated_features[4,10]
   + A[6,4] * Conv_2_aggregated_features[4,10]
   + A[7,4] * Conv_2_aggregated_features[4,10] ] = 0
 Conv_2_averages_constraint[4,11]: [
   A[0,4] * Conv_2_aggregated_features[4,11]
   + A[1,4] * Conv_2_aggregated_features[4,11]
   + A[2,4] * Conv_2_aggregated_features[4,11]
   + A[3,4] * Conv_2_aggregated_features[4,11]
   - A[4,0] * Conv_1_Relu_ts[0,11] - A[4,1] * Conv_1_Relu_ts[1,11]
   - A[4,2] * Conv_1_Relu_ts[2,11] - A[4,3] * Conv_1_Relu_ts[3,11]
   - A[4,4] * Conv_1_Relu_ts[4,11]
   + A[4,4] * Conv_2_aggregated_features[4,11]
   - A[4,5] * Conv_1_Relu_ts[5,11] - A[4,6] * Conv_1_Relu_ts[6,11]
   - A[4,7] * Conv_1_Relu_ts[7,11]
   + A[5,4] * Conv_2_aggregated_features[4,11]
   + A[6,4] * Conv_2_aggregated_features[4,11]
   + A[7,4] * Conv_2_aggregated_features[4,11] ] = 0
 Conv_2_averages_constraint[4,12]: [
   A[0,4] * Conv_2_aggregated_features[4,12]
   + A[1,4] * Conv_2_aggregated_features[4,12]
   + A[2,4] * Conv_2_aggregated_features[4,12]
   + A[3,4] * Conv_2_aggregated_features[4,12]
   - A[4,0] * Conv_1_Relu_ts[0,12] - A[4,1] * Conv_1_Relu_ts[1,12]
   - A[4,2] * Conv_1_Relu_ts[2,12] - A[4,3] * Conv_1_Relu_ts[3,12]
   - A[4,4] * Conv_1_Relu_ts[4,12]
   + A[4,4] * Conv_2_aggregated_features[4,12]
   - A[4,5] * Conv_1_Relu_ts[5,12] - A[4,6] * Conv_1_Relu_ts[6,12]
   - A[4,7] * Conv_1_Relu_ts[7,12]
   + A[5,4] * Conv_2_aggregated_features[4,12]
   + A[6,4] * Conv_2_aggregated_features[4,12]
   + A[7,4] * Conv_2_aggregated_features[4,12] ] = 0
 Conv_2_averages_constraint[4,13]: [
   A[0,4] * Conv_2_aggregated_features[4,13]
   + A[1,4] * Conv_2_aggregated_features[4,13]
   + A[2,4] * Conv_2_aggregated_features[4,13]
   + A[3,4] * Conv_2_aggregated_features[4,13]
   - A[4,0] * Conv_1_Relu_ts[0,13] - A[4,1] * Conv_1_Relu_ts[1,13]
   - A[4,2] * Conv_1_Relu_ts[2,13] - A[4,3] * Conv_1_Relu_ts[3,13]
   - A[4,4] * Conv_1_Relu_ts[4,13]
   + A[4,4] * Conv_2_aggregated_features[4,13]
   - A[4,5] * Conv_1_Relu_ts[5,13] - A[4,6] * Conv_1_Relu_ts[6,13]
   - A[4,7] * Conv_1_Relu_ts[7,13]
   + A[5,4] * Conv_2_aggregated_features[4,13]
   + A[6,4] * Conv_2_aggregated_features[4,13]
   + A[7,4] * Conv_2_aggregated_features[4,13] ] = 0
 Conv_2_averages_constraint[4,14]: [
   A[0,4] * Conv_2_aggregated_features[4,14]
   + A[1,4] * Conv_2_aggregated_features[4,14]
   + A[2,4] * Conv_2_aggregated_features[4,14]
   + A[3,4] * Conv_2_aggregated_features[4,14]
   - A[4,0] * Conv_1_Relu_ts[0,14] - A[4,1] * Conv_1_Relu_ts[1,14]
   - A[4,2] * Conv_1_Relu_ts[2,14] - A[4,3] * Conv_1_Relu_ts[3,14]
   - A[4,4] * Conv_1_Relu_ts[4,14]
   + A[4,4] * Conv_2_aggregated_features[4,14]
   - A[4,5] * Conv_1_Relu_ts[5,14] - A[4,6] * Conv_1_Relu_ts[6,14]
   - A[4,7] * Conv_1_Relu_ts[7,14]
   + A[5,4] * Conv_2_aggregated_features[4,14]
   + A[6,4] * Conv_2_aggregated_features[4,14]
   + A[7,4] * Conv_2_aggregated_features[4,14] ] = 0
 Conv_2_averages_constraint[4,15]: [
   A[0,4] * Conv_2_aggregated_features[4,15]
   + A[1,4] * Conv_2_aggregated_features[4,15]
   + A[2,4] * Conv_2_aggregated_features[4,15]
   + A[3,4] * Conv_2_aggregated_features[4,15]
   - A[4,0] * Conv_1_Relu_ts[0,15] - A[4,1] * Conv_1_Relu_ts[1,15]
   - A[4,2] * Conv_1_Relu_ts[2,15] - A[4,3] * Conv_1_Relu_ts[3,15]
   - A[4,4] * Conv_1_Relu_ts[4,15]
   + A[4,4] * Conv_2_aggregated_features[4,15]
   - A[4,5] * Conv_1_Relu_ts[5,15] - A[4,6] * Conv_1_Relu_ts[6,15]
   - A[4,7] * Conv_1_Relu_ts[7,15]
   + A[5,4] * Conv_2_aggregated_features[4,15]
   + A[6,4] * Conv_2_aggregated_features[4,15]
   + A[7,4] * Conv_2_aggregated_features[4,15] ] = 0
 Conv_2_averages_constraint[5,0]: [
   A[0,5] * Conv_2_aggregated_features[5,0]
   + A[1,5] * Conv_2_aggregated_features[5,0]
   + A[2,5] * Conv_2_aggregated_features[5,0]
   + A[3,5] * Conv_2_aggregated_features[5,0]
   + A[4,5] * Conv_2_aggregated_features[5,0]
   - A[5,0] * Conv_1_Relu_ts[0,0] - A[5,1] * Conv_1_Relu_ts[1,0]
   - A[5,2] * Conv_1_Relu_ts[2,0] - A[5,3] * Conv_1_Relu_ts[3,0]
   - A[5,4] * Conv_1_Relu_ts[4,0] - A[5,5] * Conv_1_Relu_ts[5,0]
   + A[5,5] * Conv_2_aggregated_features[5,0]
   - A[5,6] * Conv_1_Relu_ts[6,0] - A[5,7] * Conv_1_Relu_ts[7,0]
   + A[6,5] * Conv_2_aggregated_features[5,0]
   + A[7,5] * Conv_2_aggregated_features[5,0] ] = 0
 Conv_2_averages_constraint[5,1]: [
   A[0,5] * Conv_2_aggregated_features[5,1]
   + A[1,5] * Conv_2_aggregated_features[5,1]
   + A[2,5] * Conv_2_aggregated_features[5,1]
   + A[3,5] * Conv_2_aggregated_features[5,1]
   + A[4,5] * Conv_2_aggregated_features[5,1]
   - A[5,0] * Conv_1_Relu_ts[0,1] - A[5,1] * Conv_1_Relu_ts[1,1]
   - A[5,2] * Conv_1_Relu_ts[2,1] - A[5,3] * Conv_1_Relu_ts[3,1]
   - A[5,4] * Conv_1_Relu_ts[4,1] - A[5,5] * Conv_1_Relu_ts[5,1]
   + A[5,5] * Conv_2_aggregated_features[5,1]
   - A[5,6] * Conv_1_Relu_ts[6,1] - A[5,7] * Conv_1_Relu_ts[7,1]
   + A[6,5] * Conv_2_aggregated_features[5,1]
   + A[7,5] * Conv_2_aggregated_features[5,1] ] = 0
 Conv_2_averages_constraint[5,2]: [
   A[0,5] * Conv_2_aggregated_features[5,2]
   + A[1,5] * Conv_2_aggregated_features[5,2]
   + A[2,5] * Conv_2_aggregated_features[5,2]
   + A[3,5] * Conv_2_aggregated_features[5,2]
   + A[4,5] * Conv_2_aggregated_features[5,2]
   - A[5,0] * Conv_1_Relu_ts[0,2] - A[5,1] * Conv_1_Relu_ts[1,2]
   - A[5,2] * Conv_1_Relu_ts[2,2] - A[5,3] * Conv_1_Relu_ts[3,2]
   - A[5,4] * Conv_1_Relu_ts[4,2] - A[5,5] * Conv_1_Relu_ts[5,2]
   + A[5,5] * Conv_2_aggregated_features[5,2]
   - A[5,6] * Conv_1_Relu_ts[6,2] - A[5,7] * Conv_1_Relu_ts[7,2]
   + A[6,5] * Conv_2_aggregated_features[5,2]
   + A[7,5] * Conv_2_aggregated_features[5,2] ] = 0
 Conv_2_averages_constraint[5,3]: [
   A[0,5] * Conv_2_aggregated_features[5,3]
   + A[1,5] * Conv_2_aggregated_features[5,3]
   + A[2,5] * Conv_2_aggregated_features[5,3]
   + A[3,5] * Conv_2_aggregated_features[5,3]
   + A[4,5] * Conv_2_aggregated_features[5,3]
   - A[5,0] * Conv_1_Relu_ts[0,3] - A[5,1] * Conv_1_Relu_ts[1,3]
   - A[5,2] * Conv_1_Relu_ts[2,3] - A[5,3] * Conv_1_Relu_ts[3,3]
   - A[5,4] * Conv_1_Relu_ts[4,3] - A[5,5] * Conv_1_Relu_ts[5,3]
   + A[5,5] * Conv_2_aggregated_features[5,3]
   - A[5,6] * Conv_1_Relu_ts[6,3] - A[5,7] * Conv_1_Relu_ts[7,3]
   + A[6,5] * Conv_2_aggregated_features[5,3]
   + A[7,5] * Conv_2_aggregated_features[5,3] ] = 0
 Conv_2_averages_constraint[5,4]: [
   A[0,5] * Conv_2_aggregated_features[5,4]
   + A[1,5] * Conv_2_aggregated_features[5,4]
   + A[2,5] * Conv_2_aggregated_features[5,4]
   + A[3,5] * Conv_2_aggregated_features[5,4]
   + A[4,5] * Conv_2_aggregated_features[5,4]
   - A[5,0] * Conv_1_Relu_ts[0,4] - A[5,1] * Conv_1_Relu_ts[1,4]
   - A[5,2] * Conv_1_Relu_ts[2,4] - A[5,3] * Conv_1_Relu_ts[3,4]
   - A[5,4] * Conv_1_Relu_ts[4,4] - A[5,5] * Conv_1_Relu_ts[5,4]
   + A[5,5] * Conv_2_aggregated_features[5,4]
   - A[5,6] * Conv_1_Relu_ts[6,4] - A[5,7] * Conv_1_Relu_ts[7,4]
   + A[6,5] * Conv_2_aggregated_features[5,4]
   + A[7,5] * Conv_2_aggregated_features[5,4] ] = 0
 Conv_2_averages_constraint[5,5]: [
   A[0,5] * Conv_2_aggregated_features[5,5]
   + A[1,5] * Conv_2_aggregated_features[5,5]
   + A[2,5] * Conv_2_aggregated_features[5,5]
   + A[3,5] * Conv_2_aggregated_features[5,5]
   + A[4,5] * Conv_2_aggregated_features[5,5]
   - A[5,0] * Conv_1_Relu_ts[0,5] - A[5,1] * Conv_1_Relu_ts[1,5]
   - A[5,2] * Conv_1_Relu_ts[2,5] - A[5,3] * Conv_1_Relu_ts[3,5]
   - A[5,4] * Conv_1_Relu_ts[4,5] - A[5,5] * Conv_1_Relu_ts[5,5]
   + A[5,5] * Conv_2_aggregated_features[5,5]
   - A[5,6] * Conv_1_Relu_ts[6,5] - A[5,7] * Conv_1_Relu_ts[7,5]
   + A[6,5] * Conv_2_aggregated_features[5,5]
   + A[7,5] * Conv_2_aggregated_features[5,5] ] = 0
 Conv_2_averages_constraint[5,6]: [
   A[0,5] * Conv_2_aggregated_features[5,6]
   + A[1,5] * Conv_2_aggregated_features[5,6]
   + A[2,5] * Conv_2_aggregated_features[5,6]
   + A[3,5] * Conv_2_aggregated_features[5,6]
   + A[4,5] * Conv_2_aggregated_features[5,6]
   - A[5,0] * Conv_1_Relu_ts[0,6] - A[5,1] * Conv_1_Relu_ts[1,6]
   - A[5,2] * Conv_1_Relu_ts[2,6] - A[5,3] * Conv_1_Relu_ts[3,6]
   - A[5,4] * Conv_1_Relu_ts[4,6] - A[5,5] * Conv_1_Relu_ts[5,6]
   + A[5,5] * Conv_2_aggregated_features[5,6]
   - A[5,6] * Conv_1_Relu_ts[6,6] - A[5,7] * Conv_1_Relu_ts[7,6]
   + A[6,5] * Conv_2_aggregated_features[5,6]
   + A[7,5] * Conv_2_aggregated_features[5,6] ] = 0
 Conv_2_averages_constraint[5,7]: [
   A[0,5] * Conv_2_aggregated_features[5,7]
   + A[1,5] * Conv_2_aggregated_features[5,7]
   + A[2,5] * Conv_2_aggregated_features[5,7]
   + A[3,5] * Conv_2_aggregated_features[5,7]
   + A[4,5] * Conv_2_aggregated_features[5,7]
   - A[5,0] * Conv_1_Relu_ts[0,7] - A[5,1] * Conv_1_Relu_ts[1,7]
   - A[5,2] * Conv_1_Relu_ts[2,7] - A[5,3] * Conv_1_Relu_ts[3,7]
   - A[5,4] * Conv_1_Relu_ts[4,7] - A[5,5] * Conv_1_Relu_ts[5,7]
   + A[5,5] * Conv_2_aggregated_features[5,7]
   - A[5,6] * Conv_1_Relu_ts[6,7] - A[5,7] * Conv_1_Relu_ts[7,7]
   + A[6,5] * Conv_2_aggregated_features[5,7]
   + A[7,5] * Conv_2_aggregated_features[5,7] ] = 0
 Conv_2_averages_constraint[5,8]: [
   A[0,5] * Conv_2_aggregated_features[5,8]
   + A[1,5] * Conv_2_aggregated_features[5,8]
   + A[2,5] * Conv_2_aggregated_features[5,8]
   + A[3,5] * Conv_2_aggregated_features[5,8]
   + A[4,5] * Conv_2_aggregated_features[5,8]
   - A[5,0] * Conv_1_Relu_ts[0,8] - A[5,1] * Conv_1_Relu_ts[1,8]
   - A[5,2] * Conv_1_Relu_ts[2,8] - A[5,3] * Conv_1_Relu_ts[3,8]
   - A[5,4] * Conv_1_Relu_ts[4,8] - A[5,5] * Conv_1_Relu_ts[5,8]
   + A[5,5] * Conv_2_aggregated_features[5,8]
   - A[5,6] * Conv_1_Relu_ts[6,8] - A[5,7] * Conv_1_Relu_ts[7,8]
   + A[6,5] * Conv_2_aggregated_features[5,8]
   + A[7,5] * Conv_2_aggregated_features[5,8] ] = 0
 Conv_2_averages_constraint[5,9]: [
   A[0,5] * Conv_2_aggregated_features[5,9]
   + A[1,5] * Conv_2_aggregated_features[5,9]
   + A[2,5] * Conv_2_aggregated_features[5,9]
   + A[3,5] * Conv_2_aggregated_features[5,9]
   + A[4,5] * Conv_2_aggregated_features[5,9]
   - A[5,0] * Conv_1_Relu_ts[0,9] - A[5,1] * Conv_1_Relu_ts[1,9]
   - A[5,2] * Conv_1_Relu_ts[2,9] - A[5,3] * Conv_1_Relu_ts[3,9]
   - A[5,4] * Conv_1_Relu_ts[4,9] - A[5,5] * Conv_1_Relu_ts[5,9]
   + A[5,5] * Conv_2_aggregated_features[5,9]
   - A[5,6] * Conv_1_Relu_ts[6,9] - A[5,7] * Conv_1_Relu_ts[7,9]
   + A[6,5] * Conv_2_aggregated_features[5,9]
   + A[7,5] * Conv_2_aggregated_features[5,9] ] = 0
 Conv_2_averages_constraint[5,10]: [
   A[0,5] * Conv_2_aggregated_features[5,10]
   + A[1,5] * Conv_2_aggregated_features[5,10]
   + A[2,5] * Conv_2_aggregated_features[5,10]
   + A[3,5] * Conv_2_aggregated_features[5,10]
   + A[4,5] * Conv_2_aggregated_features[5,10]
   - A[5,0] * Conv_1_Relu_ts[0,10] - A[5,1] * Conv_1_Relu_ts[1,10]
   - A[5,2] * Conv_1_Relu_ts[2,10] - A[5,3] * Conv_1_Relu_ts[3,10]
   - A[5,4] * Conv_1_Relu_ts[4,10] - A[5,5] * Conv_1_Relu_ts[5,10]
   + A[5,5] * Conv_2_aggregated_features[5,10]
   - A[5,6] * Conv_1_Relu_ts[6,10] - A[5,7] * Conv_1_Relu_ts[7,10]
   + A[6,5] * Conv_2_aggregated_features[5,10]
   + A[7,5] * Conv_2_aggregated_features[5,10] ] = 0
 Conv_2_averages_constraint[5,11]: [
   A[0,5] * Conv_2_aggregated_features[5,11]
   + A[1,5] * Conv_2_aggregated_features[5,11]
   + A[2,5] * Conv_2_aggregated_features[5,11]
   + A[3,5] * Conv_2_aggregated_features[5,11]
   + A[4,5] * Conv_2_aggregated_features[5,11]
   - A[5,0] * Conv_1_Relu_ts[0,11] - A[5,1] * Conv_1_Relu_ts[1,11]
   - A[5,2] * Conv_1_Relu_ts[2,11] - A[5,3] * Conv_1_Relu_ts[3,11]
   - A[5,4] * Conv_1_Relu_ts[4,11] - A[5,5] * Conv_1_Relu_ts[5,11]
   + A[5,5] * Conv_2_aggregated_features[5,11]
   - A[5,6] * Conv_1_Relu_ts[6,11] - A[5,7] * Conv_1_Relu_ts[7,11]
   + A[6,5] * Conv_2_aggregated_features[5,11]
   + A[7,5] * Conv_2_aggregated_features[5,11] ] = 0
 Conv_2_averages_constraint[5,12]: [
   A[0,5] * Conv_2_aggregated_features[5,12]
   + A[1,5] * Conv_2_aggregated_features[5,12]
   + A[2,5] * Conv_2_aggregated_features[5,12]
   + A[3,5] * Conv_2_aggregated_features[5,12]
   + A[4,5] * Conv_2_aggregated_features[5,12]
   - A[5,0] * Conv_1_Relu_ts[0,12] - A[5,1] * Conv_1_Relu_ts[1,12]
   - A[5,2] * Conv_1_Relu_ts[2,12] - A[5,3] * Conv_1_Relu_ts[3,12]
   - A[5,4] * Conv_1_Relu_ts[4,12] - A[5,5] * Conv_1_Relu_ts[5,12]
   + A[5,5] * Conv_2_aggregated_features[5,12]
   - A[5,6] * Conv_1_Relu_ts[6,12] - A[5,7] * Conv_1_Relu_ts[7,12]
   + A[6,5] * Conv_2_aggregated_features[5,12]
   + A[7,5] * Conv_2_aggregated_features[5,12] ] = 0
 Conv_2_averages_constraint[5,13]: [
   A[0,5] * Conv_2_aggregated_features[5,13]
   + A[1,5] * Conv_2_aggregated_features[5,13]
   + A[2,5] * Conv_2_aggregated_features[5,13]
   + A[3,5] * Conv_2_aggregated_features[5,13]
   + A[4,5] * Conv_2_aggregated_features[5,13]
   - A[5,0] * Conv_1_Relu_ts[0,13] - A[5,1] * Conv_1_Relu_ts[1,13]
   - A[5,2] * Conv_1_Relu_ts[2,13] - A[5,3] * Conv_1_Relu_ts[3,13]
   - A[5,4] * Conv_1_Relu_ts[4,13] - A[5,5] * Conv_1_Relu_ts[5,13]
   + A[5,5] * Conv_2_aggregated_features[5,13]
   - A[5,6] * Conv_1_Relu_ts[6,13] - A[5,7] * Conv_1_Relu_ts[7,13]
   + A[6,5] * Conv_2_aggregated_features[5,13]
   + A[7,5] * Conv_2_aggregated_features[5,13] ] = 0
 Conv_2_averages_constraint[5,14]: [
   A[0,5] * Conv_2_aggregated_features[5,14]
   + A[1,5] * Conv_2_aggregated_features[5,14]
   + A[2,5] * Conv_2_aggregated_features[5,14]
   + A[3,5] * Conv_2_aggregated_features[5,14]
   + A[4,5] * Conv_2_aggregated_features[5,14]
   - A[5,0] * Conv_1_Relu_ts[0,14] - A[5,1] * Conv_1_Relu_ts[1,14]
   - A[5,2] * Conv_1_Relu_ts[2,14] - A[5,3] * Conv_1_Relu_ts[3,14]
   - A[5,4] * Conv_1_Relu_ts[4,14] - A[5,5] * Conv_1_Relu_ts[5,14]
   + A[5,5] * Conv_2_aggregated_features[5,14]
   - A[5,6] * Conv_1_Relu_ts[6,14] - A[5,7] * Conv_1_Relu_ts[7,14]
   + A[6,5] * Conv_2_aggregated_features[5,14]
   + A[7,5] * Conv_2_aggregated_features[5,14] ] = 0
 Conv_2_averages_constraint[5,15]: [
   A[0,5] * Conv_2_aggregated_features[5,15]
   + A[1,5] * Conv_2_aggregated_features[5,15]
   + A[2,5] * Conv_2_aggregated_features[5,15]
   + A[3,5] * Conv_2_aggregated_features[5,15]
   + A[4,5] * Conv_2_aggregated_features[5,15]
   - A[5,0] * Conv_1_Relu_ts[0,15] - A[5,1] * Conv_1_Relu_ts[1,15]
   - A[5,2] * Conv_1_Relu_ts[2,15] - A[5,3] * Conv_1_Relu_ts[3,15]
   - A[5,4] * Conv_1_Relu_ts[4,15] - A[5,5] * Conv_1_Relu_ts[5,15]
   + A[5,5] * Conv_2_aggregated_features[5,15]
   - A[5,6] * Conv_1_Relu_ts[6,15] - A[5,7] * Conv_1_Relu_ts[7,15]
   + A[6,5] * Conv_2_aggregated_features[5,15]
   + A[7,5] * Conv_2_aggregated_features[5,15] ] = 0
 Conv_2_averages_constraint[6,0]: [
   A[0,6] * Conv_2_aggregated_features[6,0]
   + A[1,6] * Conv_2_aggregated_features[6,0]
   + A[2,6] * Conv_2_aggregated_features[6,0]
   + A[3,6] * Conv_2_aggregated_features[6,0]
   + A[4,6] * Conv_2_aggregated_features[6,0]
   + A[5,6] * Conv_2_aggregated_features[6,0]
   - A[6,0] * Conv_1_Relu_ts[0,0] - A[6,1] * Conv_1_Relu_ts[1,0]
   - A[6,2] * Conv_1_Relu_ts[2,0] - A[6,3] * Conv_1_Relu_ts[3,0]
   - A[6,4] * Conv_1_Relu_ts[4,0] - A[6,5] * Conv_1_Relu_ts[5,0]
   - A[6,6] * Conv_1_Relu_ts[6,0]
   + A[6,6] * Conv_2_aggregated_features[6,0]
   - A[6,7] * Conv_1_Relu_ts[7,0]
   + A[7,6] * Conv_2_aggregated_features[6,0] ] = 0
 Conv_2_averages_constraint[6,1]: [
   A[0,6] * Conv_2_aggregated_features[6,1]
   + A[1,6] * Conv_2_aggregated_features[6,1]
   + A[2,6] * Conv_2_aggregated_features[6,1]
   + A[3,6] * Conv_2_aggregated_features[6,1]
   + A[4,6] * Conv_2_aggregated_features[6,1]
   + A[5,6] * Conv_2_aggregated_features[6,1]
   - A[6,0] * Conv_1_Relu_ts[0,1] - A[6,1] * Conv_1_Relu_ts[1,1]
   - A[6,2] * Conv_1_Relu_ts[2,1] - A[6,3] * Conv_1_Relu_ts[3,1]
   - A[6,4] * Conv_1_Relu_ts[4,1] - A[6,5] * Conv_1_Relu_ts[5,1]
   - A[6,6] * Conv_1_Relu_ts[6,1]
   + A[6,6] * Conv_2_aggregated_features[6,1]
   - A[6,7] * Conv_1_Relu_ts[7,1]
   + A[7,6] * Conv_2_aggregated_features[6,1] ] = 0
 Conv_2_averages_constraint[6,2]: [
   A[0,6] * Conv_2_aggregated_features[6,2]
   + A[1,6] * Conv_2_aggregated_features[6,2]
   + A[2,6] * Conv_2_aggregated_features[6,2]
   + A[3,6] * Conv_2_aggregated_features[6,2]
   + A[4,6] * Conv_2_aggregated_features[6,2]
   + A[5,6] * Conv_2_aggregated_features[6,2]
   - A[6,0] * Conv_1_Relu_ts[0,2] - A[6,1] * Conv_1_Relu_ts[1,2]
   - A[6,2] * Conv_1_Relu_ts[2,2] - A[6,3] * Conv_1_Relu_ts[3,2]
   - A[6,4] * Conv_1_Relu_ts[4,2] - A[6,5] * Conv_1_Relu_ts[5,2]
   - A[6,6] * Conv_1_Relu_ts[6,2]
   + A[6,6] * Conv_2_aggregated_features[6,2]
   - A[6,7] * Conv_1_Relu_ts[7,2]
   + A[7,6] * Conv_2_aggregated_features[6,2] ] = 0
 Conv_2_averages_constraint[6,3]: [
   A[0,6] * Conv_2_aggregated_features[6,3]
   + A[1,6] * Conv_2_aggregated_features[6,3]
   + A[2,6] * Conv_2_aggregated_features[6,3]
   + A[3,6] * Conv_2_aggregated_features[6,3]
   + A[4,6] * Conv_2_aggregated_features[6,3]
   + A[5,6] * Conv_2_aggregated_features[6,3]
   - A[6,0] * Conv_1_Relu_ts[0,3] - A[6,1] * Conv_1_Relu_ts[1,3]
   - A[6,2] * Conv_1_Relu_ts[2,3] - A[6,3] * Conv_1_Relu_ts[3,3]
   - A[6,4] * Conv_1_Relu_ts[4,3] - A[6,5] * Conv_1_Relu_ts[5,3]
   - A[6,6] * Conv_1_Relu_ts[6,3]
   + A[6,6] * Conv_2_aggregated_features[6,3]
   - A[6,7] * Conv_1_Relu_ts[7,3]
   + A[7,6] * Conv_2_aggregated_features[6,3] ] = 0
 Conv_2_averages_constraint[6,4]: [
   A[0,6] * Conv_2_aggregated_features[6,4]
   + A[1,6] * Conv_2_aggregated_features[6,4]
   + A[2,6] * Conv_2_aggregated_features[6,4]
   + A[3,6] * Conv_2_aggregated_features[6,4]
   + A[4,6] * Conv_2_aggregated_features[6,4]
   + A[5,6] * Conv_2_aggregated_features[6,4]
   - A[6,0] * Conv_1_Relu_ts[0,4] - A[6,1] * Conv_1_Relu_ts[1,4]
   - A[6,2] * Conv_1_Relu_ts[2,4] - A[6,3] * Conv_1_Relu_ts[3,4]
   - A[6,4] * Conv_1_Relu_ts[4,4] - A[6,5] * Conv_1_Relu_ts[5,4]
   - A[6,6] * Conv_1_Relu_ts[6,4]
   + A[6,6] * Conv_2_aggregated_features[6,4]
   - A[6,7] * Conv_1_Relu_ts[7,4]
   + A[7,6] * Conv_2_aggregated_features[6,4] ] = 0
 Conv_2_averages_constraint[6,5]: [
   A[0,6] * Conv_2_aggregated_features[6,5]
   + A[1,6] * Conv_2_aggregated_features[6,5]
   + A[2,6] * Conv_2_aggregated_features[6,5]
   + A[3,6] * Conv_2_aggregated_features[6,5]
   + A[4,6] * Conv_2_aggregated_features[6,5]
   + A[5,6] * Conv_2_aggregated_features[6,5]
   - A[6,0] * Conv_1_Relu_ts[0,5] - A[6,1] * Conv_1_Relu_ts[1,5]
   - A[6,2] * Conv_1_Relu_ts[2,5] - A[6,3] * Conv_1_Relu_ts[3,5]
   - A[6,4] * Conv_1_Relu_ts[4,5] - A[6,5] * Conv_1_Relu_ts[5,5]
   - A[6,6] * Conv_1_Relu_ts[6,5]
   + A[6,6] * Conv_2_aggregated_features[6,5]
   - A[6,7] * Conv_1_Relu_ts[7,5]
   + A[7,6] * Conv_2_aggregated_features[6,5] ] = 0
 Conv_2_averages_constraint[6,6]: [
   A[0,6] * Conv_2_aggregated_features[6,6]
   + A[1,6] * Conv_2_aggregated_features[6,6]
   + A[2,6] * Conv_2_aggregated_features[6,6]
   + A[3,6] * Conv_2_aggregated_features[6,6]
   + A[4,6] * Conv_2_aggregated_features[6,6]
   + A[5,6] * Conv_2_aggregated_features[6,6]
   - A[6,0] * Conv_1_Relu_ts[0,6] - A[6,1] * Conv_1_Relu_ts[1,6]
   - A[6,2] * Conv_1_Relu_ts[2,6] - A[6,3] * Conv_1_Relu_ts[3,6]
   - A[6,4] * Conv_1_Relu_ts[4,6] - A[6,5] * Conv_1_Relu_ts[5,6]
   - A[6,6] * Conv_1_Relu_ts[6,6]
   + A[6,6] * Conv_2_aggregated_features[6,6]
   - A[6,7] * Conv_1_Relu_ts[7,6]
   + A[7,6] * Conv_2_aggregated_features[6,6] ] = 0
 Conv_2_averages_constraint[6,7]: [
   A[0,6] * Conv_2_aggregated_features[6,7]
   + A[1,6] * Conv_2_aggregated_features[6,7]
   + A[2,6] * Conv_2_aggregated_features[6,7]
   + A[3,6] * Conv_2_aggregated_features[6,7]
   + A[4,6] * Conv_2_aggregated_features[6,7]
   + A[5,6] * Conv_2_aggregated_features[6,7]
   - A[6,0] * Conv_1_Relu_ts[0,7] - A[6,1] * Conv_1_Relu_ts[1,7]
   - A[6,2] * Conv_1_Relu_ts[2,7] - A[6,3] * Conv_1_Relu_ts[3,7]
   - A[6,4] * Conv_1_Relu_ts[4,7] - A[6,5] * Conv_1_Relu_ts[5,7]
   - A[6,6] * Conv_1_Relu_ts[6,7]
   + A[6,6] * Conv_2_aggregated_features[6,7]
   - A[6,7] * Conv_1_Relu_ts[7,7]
   + A[7,6] * Conv_2_aggregated_features[6,7] ] = 0
 Conv_2_averages_constraint[6,8]: [
   A[0,6] * Conv_2_aggregated_features[6,8]
   + A[1,6] * Conv_2_aggregated_features[6,8]
   + A[2,6] * Conv_2_aggregated_features[6,8]
   + A[3,6] * Conv_2_aggregated_features[6,8]
   + A[4,6] * Conv_2_aggregated_features[6,8]
   + A[5,6] * Conv_2_aggregated_features[6,8]
   - A[6,0] * Conv_1_Relu_ts[0,8] - A[6,1] * Conv_1_Relu_ts[1,8]
   - A[6,2] * Conv_1_Relu_ts[2,8] - A[6,3] * Conv_1_Relu_ts[3,8]
   - A[6,4] * Conv_1_Relu_ts[4,8] - A[6,5] * Conv_1_Relu_ts[5,8]
   - A[6,6] * Conv_1_Relu_ts[6,8]
   + A[6,6] * Conv_2_aggregated_features[6,8]
   - A[6,7] * Conv_1_Relu_ts[7,8]
   + A[7,6] * Conv_2_aggregated_features[6,8] ] = 0
 Conv_2_averages_constraint[6,9]: [
   A[0,6] * Conv_2_aggregated_features[6,9]
   + A[1,6] * Conv_2_aggregated_features[6,9]
   + A[2,6] * Conv_2_aggregated_features[6,9]
   + A[3,6] * Conv_2_aggregated_features[6,9]
   + A[4,6] * Conv_2_aggregated_features[6,9]
   + A[5,6] * Conv_2_aggregated_features[6,9]
   - A[6,0] * Conv_1_Relu_ts[0,9] - A[6,1] * Conv_1_Relu_ts[1,9]
   - A[6,2] * Conv_1_Relu_ts[2,9] - A[6,3] * Conv_1_Relu_ts[3,9]
   - A[6,4] * Conv_1_Relu_ts[4,9] - A[6,5] * Conv_1_Relu_ts[5,9]
   - A[6,6] * Conv_1_Relu_ts[6,9]
   + A[6,6] * Conv_2_aggregated_features[6,9]
   - A[6,7] * Conv_1_Relu_ts[7,9]
   + A[7,6] * Conv_2_aggregated_features[6,9] ] = 0
 Conv_2_averages_constraint[6,10]: [
   A[0,6] * Conv_2_aggregated_features[6,10]
   + A[1,6] * Conv_2_aggregated_features[6,10]
   + A[2,6] * Conv_2_aggregated_features[6,10]
   + A[3,6] * Conv_2_aggregated_features[6,10]
   + A[4,6] * Conv_2_aggregated_features[6,10]
   + A[5,6] * Conv_2_aggregated_features[6,10]
   - A[6,0] * Conv_1_Relu_ts[0,10] - A[6,1] * Conv_1_Relu_ts[1,10]
   - A[6,2] * Conv_1_Relu_ts[2,10] - A[6,3] * Conv_1_Relu_ts[3,10]
   - A[6,4] * Conv_1_Relu_ts[4,10] - A[6,5] * Conv_1_Relu_ts[5,10]
   - A[6,6] * Conv_1_Relu_ts[6,10]
   + A[6,6] * Conv_2_aggregated_features[6,10]
   - A[6,7] * Conv_1_Relu_ts[7,10]
   + A[7,6] * Conv_2_aggregated_features[6,10] ] = 0
 Conv_2_averages_constraint[6,11]: [
   A[0,6] * Conv_2_aggregated_features[6,11]
   + A[1,6] * Conv_2_aggregated_features[6,11]
   + A[2,6] * Conv_2_aggregated_features[6,11]
   + A[3,6] * Conv_2_aggregated_features[6,11]
   + A[4,6] * Conv_2_aggregated_features[6,11]
   + A[5,6] * Conv_2_aggregated_features[6,11]
   - A[6,0] * Conv_1_Relu_ts[0,11] - A[6,1] * Conv_1_Relu_ts[1,11]
   - A[6,2] * Conv_1_Relu_ts[2,11] - A[6,3] * Conv_1_Relu_ts[3,11]
   - A[6,4] * Conv_1_Relu_ts[4,11] - A[6,5] * Conv_1_Relu_ts[5,11]
   - A[6,6] * Conv_1_Relu_ts[6,11]
   + A[6,6] * Conv_2_aggregated_features[6,11]
   - A[6,7] * Conv_1_Relu_ts[7,11]
   + A[7,6] * Conv_2_aggregated_features[6,11] ] = 0
 Conv_2_averages_constraint[6,12]: [
   A[0,6] * Conv_2_aggregated_features[6,12]
   + A[1,6] * Conv_2_aggregated_features[6,12]
   + A[2,6] * Conv_2_aggregated_features[6,12]
   + A[3,6] * Conv_2_aggregated_features[6,12]
   + A[4,6] * Conv_2_aggregated_features[6,12]
   + A[5,6] * Conv_2_aggregated_features[6,12]
   - A[6,0] * Conv_1_Relu_ts[0,12] - A[6,1] * Conv_1_Relu_ts[1,12]
   - A[6,2] * Conv_1_Relu_ts[2,12] - A[6,3] * Conv_1_Relu_ts[3,12]
   - A[6,4] * Conv_1_Relu_ts[4,12] - A[6,5] * Conv_1_Relu_ts[5,12]
   - A[6,6] * Conv_1_Relu_ts[6,12]
   + A[6,6] * Conv_2_aggregated_features[6,12]
   - A[6,7] * Conv_1_Relu_ts[7,12]
   + A[7,6] * Conv_2_aggregated_features[6,12] ] = 0
 Conv_2_averages_constraint[6,13]: [
   A[0,6] * Conv_2_aggregated_features[6,13]
   + A[1,6] * Conv_2_aggregated_features[6,13]
   + A[2,6] * Conv_2_aggregated_features[6,13]
   + A[3,6] * Conv_2_aggregated_features[6,13]
   + A[4,6] * Conv_2_aggregated_features[6,13]
   + A[5,6] * Conv_2_aggregated_features[6,13]
   - A[6,0] * Conv_1_Relu_ts[0,13] - A[6,1] * Conv_1_Relu_ts[1,13]
   - A[6,2] * Conv_1_Relu_ts[2,13] - A[6,3] * Conv_1_Relu_ts[3,13]
   - A[6,4] * Conv_1_Relu_ts[4,13] - A[6,5] * Conv_1_Relu_ts[5,13]
   - A[6,6] * Conv_1_Relu_ts[6,13]
   + A[6,6] * Conv_2_aggregated_features[6,13]
   - A[6,7] * Conv_1_Relu_ts[7,13]
   + A[7,6] * Conv_2_aggregated_features[6,13] ] = 0
 Conv_2_averages_constraint[6,14]: [
   A[0,6] * Conv_2_aggregated_features[6,14]
   + A[1,6] * Conv_2_aggregated_features[6,14]
   + A[2,6] * Conv_2_aggregated_features[6,14]
   + A[3,6] * Conv_2_aggregated_features[6,14]
   + A[4,6] * Conv_2_aggregated_features[6,14]
   + A[5,6] * Conv_2_aggregated_features[6,14]
   - A[6,0] * Conv_1_Relu_ts[0,14] - A[6,1] * Conv_1_Relu_ts[1,14]
   - A[6,2] * Conv_1_Relu_ts[2,14] - A[6,3] * Conv_1_Relu_ts[3,14]
   - A[6,4] * Conv_1_Relu_ts[4,14] - A[6,5] * Conv_1_Relu_ts[5,14]
   - A[6,6] * Conv_1_Relu_ts[6,14]
   + A[6,6] * Conv_2_aggregated_features[6,14]
   - A[6,7] * Conv_1_Relu_ts[7,14]
   + A[7,6] * Conv_2_aggregated_features[6,14] ] = 0
 Conv_2_averages_constraint[6,15]: [
   A[0,6] * Conv_2_aggregated_features[6,15]
   + A[1,6] * Conv_2_aggregated_features[6,15]
   + A[2,6] * Conv_2_aggregated_features[6,15]
   + A[3,6] * Conv_2_aggregated_features[6,15]
   + A[4,6] * Conv_2_aggregated_features[6,15]
   + A[5,6] * Conv_2_aggregated_features[6,15]
   - A[6,0] * Conv_1_Relu_ts[0,15] - A[6,1] * Conv_1_Relu_ts[1,15]
   - A[6,2] * Conv_1_Relu_ts[2,15] - A[6,3] * Conv_1_Relu_ts[3,15]
   - A[6,4] * Conv_1_Relu_ts[4,15] - A[6,5] * Conv_1_Relu_ts[5,15]
   - A[6,6] * Conv_1_Relu_ts[6,15]
   + A[6,6] * Conv_2_aggregated_features[6,15]
   - A[6,7] * Conv_1_Relu_ts[7,15]
   + A[7,6] * Conv_2_aggregated_features[6,15] ] = 0
 Conv_2_averages_constraint[7,0]: [
   A[0,7] * Conv_2_aggregated_features[7,0]
   + A[1,7] * Conv_2_aggregated_features[7,0]
   + A[2,7] * Conv_2_aggregated_features[7,0]
   + A[3,7] * Conv_2_aggregated_features[7,0]
   + A[4,7] * Conv_2_aggregated_features[7,0]
   + A[5,7] * Conv_2_aggregated_features[7,0]
   + A[6,7] * Conv_2_aggregated_features[7,0]
   - A[7,0] * Conv_1_Relu_ts[0,0] - A[7,1] * Conv_1_Relu_ts[1,0]
   - A[7,2] * Conv_1_Relu_ts[2,0] - A[7,3] * Conv_1_Relu_ts[3,0]
   - A[7,4] * Conv_1_Relu_ts[4,0] - A[7,5] * Conv_1_Relu_ts[5,0]
   - A[7,6] * Conv_1_Relu_ts[6,0] - A[7,7] * Conv_1_Relu_ts[7,0]
   + A[7,7] * Conv_2_aggregated_features[7,0] ] = 0
 Conv_2_averages_constraint[7,1]: [
   A[0,7] * Conv_2_aggregated_features[7,1]
   + A[1,7] * Conv_2_aggregated_features[7,1]
   + A[2,7] * Conv_2_aggregated_features[7,1]
   + A[3,7] * Conv_2_aggregated_features[7,1]
   + A[4,7] * Conv_2_aggregated_features[7,1]
   + A[5,7] * Conv_2_aggregated_features[7,1]
   + A[6,7] * Conv_2_aggregated_features[7,1]
   - A[7,0] * Conv_1_Relu_ts[0,1] - A[7,1] * Conv_1_Relu_ts[1,1]
   - A[7,2] * Conv_1_Relu_ts[2,1] - A[7,3] * Conv_1_Relu_ts[3,1]
   - A[7,4] * Conv_1_Relu_ts[4,1] - A[7,5] * Conv_1_Relu_ts[5,1]
   - A[7,6] * Conv_1_Relu_ts[6,1] - A[7,7] * Conv_1_Relu_ts[7,1]
   + A[7,7] * Conv_2_aggregated_features[7,1] ] = 0
 Conv_2_averages_constraint[7,2]: [
   A[0,7] * Conv_2_aggregated_features[7,2]
   + A[1,7] * Conv_2_aggregated_features[7,2]
   + A[2,7] * Conv_2_aggregated_features[7,2]
   + A[3,7] * Conv_2_aggregated_features[7,2]
   + A[4,7] * Conv_2_aggregated_features[7,2]
   + A[5,7] * Conv_2_aggregated_features[7,2]
   + A[6,7] * Conv_2_aggregated_features[7,2]
   - A[7,0] * Conv_1_Relu_ts[0,2] - A[7,1] * Conv_1_Relu_ts[1,2]
   - A[7,2] * Conv_1_Relu_ts[2,2] - A[7,3] * Conv_1_Relu_ts[3,2]
   - A[7,4] * Conv_1_Relu_ts[4,2] - A[7,5] * Conv_1_Relu_ts[5,2]
   - A[7,6] * Conv_1_Relu_ts[6,2] - A[7,7] * Conv_1_Relu_ts[7,2]
   + A[7,7] * Conv_2_aggregated_features[7,2] ] = 0
 Conv_2_averages_constraint[7,3]: [
   A[0,7] * Conv_2_aggregated_features[7,3]
   + A[1,7] * Conv_2_aggregated_features[7,3]
   + A[2,7] * Conv_2_aggregated_features[7,3]
   + A[3,7] * Conv_2_aggregated_features[7,3]
   + A[4,7] * Conv_2_aggregated_features[7,3]
   + A[5,7] * Conv_2_aggregated_features[7,3]
   + A[6,7] * Conv_2_aggregated_features[7,3]
   - A[7,0] * Conv_1_Relu_ts[0,3] - A[7,1] * Conv_1_Relu_ts[1,3]
   - A[7,2] * Conv_1_Relu_ts[2,3] - A[7,3] * Conv_1_Relu_ts[3,3]
   - A[7,4] * Conv_1_Relu_ts[4,3] - A[7,5] * Conv_1_Relu_ts[5,3]
   - A[7,6] * Conv_1_Relu_ts[6,3] - A[7,7] * Conv_1_Relu_ts[7,3]
   + A[7,7] * Conv_2_aggregated_features[7,3] ] = 0
 Conv_2_averages_constraint[7,4]: [
   A[0,7] * Conv_2_aggregated_features[7,4]
   + A[1,7] * Conv_2_aggregated_features[7,4]
   + A[2,7] * Conv_2_aggregated_features[7,4]
   + A[3,7] * Conv_2_aggregated_features[7,4]
   + A[4,7] * Conv_2_aggregated_features[7,4]
   + A[5,7] * Conv_2_aggregated_features[7,4]
   + A[6,7] * Conv_2_aggregated_features[7,4]
   - A[7,0] * Conv_1_Relu_ts[0,4] - A[7,1] * Conv_1_Relu_ts[1,4]
   - A[7,2] * Conv_1_Relu_ts[2,4] - A[7,3] * Conv_1_Relu_ts[3,4]
   - A[7,4] * Conv_1_Relu_ts[4,4] - A[7,5] * Conv_1_Relu_ts[5,4]
   - A[7,6] * Conv_1_Relu_ts[6,4] - A[7,7] * Conv_1_Relu_ts[7,4]
   + A[7,7] * Conv_2_aggregated_features[7,4] ] = 0
 Conv_2_averages_constraint[7,5]: [
   A[0,7] * Conv_2_aggregated_features[7,5]
   + A[1,7] * Conv_2_aggregated_features[7,5]
   + A[2,7] * Conv_2_aggregated_features[7,5]
   + A[3,7] * Conv_2_aggregated_features[7,5]
   + A[4,7] * Conv_2_aggregated_features[7,5]
   + A[5,7] * Conv_2_aggregated_features[7,5]
   + A[6,7] * Conv_2_aggregated_features[7,5]
   - A[7,0] * Conv_1_Relu_ts[0,5] - A[7,1] * Conv_1_Relu_ts[1,5]
   - A[7,2] * Conv_1_Relu_ts[2,5] - A[7,3] * Conv_1_Relu_ts[3,5]
   - A[7,4] * Conv_1_Relu_ts[4,5] - A[7,5] * Conv_1_Relu_ts[5,5]
   - A[7,6] * Conv_1_Relu_ts[6,5] - A[7,7] * Conv_1_Relu_ts[7,5]
   + A[7,7] * Conv_2_aggregated_features[7,5] ] = 0
 Conv_2_averages_constraint[7,6]: [
   A[0,7] * Conv_2_aggregated_features[7,6]
   + A[1,7] * Conv_2_aggregated_features[7,6]
   + A[2,7] * Conv_2_aggregated_features[7,6]
   + A[3,7] * Conv_2_aggregated_features[7,6]
   + A[4,7] * Conv_2_aggregated_features[7,6]
   + A[5,7] * Conv_2_aggregated_features[7,6]
   + A[6,7] * Conv_2_aggregated_features[7,6]
   - A[7,0] * Conv_1_Relu_ts[0,6] - A[7,1] * Conv_1_Relu_ts[1,6]
   - A[7,2] * Conv_1_Relu_ts[2,6] - A[7,3] * Conv_1_Relu_ts[3,6]
   - A[7,4] * Conv_1_Relu_ts[4,6] - A[7,5] * Conv_1_Relu_ts[5,6]
   - A[7,6] * Conv_1_Relu_ts[6,6] - A[7,7] * Conv_1_Relu_ts[7,6]
   + A[7,7] * Conv_2_aggregated_features[7,6] ] = 0
 Conv_2_averages_constraint[7,7]: [
   A[0,7] * Conv_2_aggregated_features[7,7]
   + A[1,7] * Conv_2_aggregated_features[7,7]
   + A[2,7] * Conv_2_aggregated_features[7,7]
   + A[3,7] * Conv_2_aggregated_features[7,7]
   + A[4,7] * Conv_2_aggregated_features[7,7]
   + A[5,7] * Conv_2_aggregated_features[7,7]
   + A[6,7] * Conv_2_aggregated_features[7,7]
   - A[7,0] * Conv_1_Relu_ts[0,7] - A[7,1] * Conv_1_Relu_ts[1,7]
   - A[7,2] * Conv_1_Relu_ts[2,7] - A[7,3] * Conv_1_Relu_ts[3,7]
   - A[7,4] * Conv_1_Relu_ts[4,7] - A[7,5] * Conv_1_Relu_ts[5,7]
   - A[7,6] * Conv_1_Relu_ts[6,7] - A[7,7] * Conv_1_Relu_ts[7,7]
   + A[7,7] * Conv_2_aggregated_features[7,7] ] = 0
 Conv_2_averages_constraint[7,8]: [
   A[0,7] * Conv_2_aggregated_features[7,8]
   + A[1,7] * Conv_2_aggregated_features[7,8]
   + A[2,7] * Conv_2_aggregated_features[7,8]
   + A[3,7] * Conv_2_aggregated_features[7,8]
   + A[4,7] * Conv_2_aggregated_features[7,8]
   + A[5,7] * Conv_2_aggregated_features[7,8]
   + A[6,7] * Conv_2_aggregated_features[7,8]
   - A[7,0] * Conv_1_Relu_ts[0,8] - A[7,1] * Conv_1_Relu_ts[1,8]
   - A[7,2] * Conv_1_Relu_ts[2,8] - A[7,3] * Conv_1_Relu_ts[3,8]
   - A[7,4] * Conv_1_Relu_ts[4,8] - A[7,5] * Conv_1_Relu_ts[5,8]
   - A[7,6] * Conv_1_Relu_ts[6,8] - A[7,7] * Conv_1_Relu_ts[7,8]
   + A[7,7] * Conv_2_aggregated_features[7,8] ] = 0
 Conv_2_averages_constraint[7,9]: [
   A[0,7] * Conv_2_aggregated_features[7,9]
   + A[1,7] * Conv_2_aggregated_features[7,9]
   + A[2,7] * Conv_2_aggregated_features[7,9]
   + A[3,7] * Conv_2_aggregated_features[7,9]
   + A[4,7] * Conv_2_aggregated_features[7,9]
   + A[5,7] * Conv_2_aggregated_features[7,9]
   + A[6,7] * Conv_2_aggregated_features[7,9]
   - A[7,0] * Conv_1_Relu_ts[0,9] - A[7,1] * Conv_1_Relu_ts[1,9]
   - A[7,2] * Conv_1_Relu_ts[2,9] - A[7,3] * Conv_1_Relu_ts[3,9]
   - A[7,4] * Conv_1_Relu_ts[4,9] - A[7,5] * Conv_1_Relu_ts[5,9]
   - A[7,6] * Conv_1_Relu_ts[6,9] - A[7,7] * Conv_1_Relu_ts[7,9]
   + A[7,7] * Conv_2_aggregated_features[7,9] ] = 0
 Conv_2_averages_constraint[7,10]: [
   A[0,7] * Conv_2_aggregated_features[7,10]
   + A[1,7] * Conv_2_aggregated_features[7,10]
   + A[2,7] * Conv_2_aggregated_features[7,10]
   + A[3,7] * Conv_2_aggregated_features[7,10]
   + A[4,7] * Conv_2_aggregated_features[7,10]
   + A[5,7] * Conv_2_aggregated_features[7,10]
   + A[6,7] * Conv_2_aggregated_features[7,10]
   - A[7,0] * Conv_1_Relu_ts[0,10] - A[7,1] * Conv_1_Relu_ts[1,10]
   - A[7,2] * Conv_1_Relu_ts[2,10] - A[7,3] * Conv_1_Relu_ts[3,10]
   - A[7,4] * Conv_1_Relu_ts[4,10] - A[7,5] * Conv_1_Relu_ts[5,10]
   - A[7,6] * Conv_1_Relu_ts[6,10] - A[7,7] * Conv_1_Relu_ts[7,10]
   + A[7,7] * Conv_2_aggregated_features[7,10] ] = 0
 Conv_2_averages_constraint[7,11]: [
   A[0,7] * Conv_2_aggregated_features[7,11]
   + A[1,7] * Conv_2_aggregated_features[7,11]
   + A[2,7] * Conv_2_aggregated_features[7,11]
   + A[3,7] * Conv_2_aggregated_features[7,11]
   + A[4,7] * Conv_2_aggregated_features[7,11]
   + A[5,7] * Conv_2_aggregated_features[7,11]
   + A[6,7] * Conv_2_aggregated_features[7,11]
   - A[7,0] * Conv_1_Relu_ts[0,11] - A[7,1] * Conv_1_Relu_ts[1,11]
   - A[7,2] * Conv_1_Relu_ts[2,11] - A[7,3] * Conv_1_Relu_ts[3,11]
   - A[7,4] * Conv_1_Relu_ts[4,11] - A[7,5] * Conv_1_Relu_ts[5,11]
   - A[7,6] * Conv_1_Relu_ts[6,11] - A[7,7] * Conv_1_Relu_ts[7,11]
   + A[7,7] * Conv_2_aggregated_features[7,11] ] = 0
 Conv_2_averages_constraint[7,12]: [
   A[0,7] * Conv_2_aggregated_features[7,12]
   + A[1,7] * Conv_2_aggregated_features[7,12]
   + A[2,7] * Conv_2_aggregated_features[7,12]
   + A[3,7] * Conv_2_aggregated_features[7,12]
   + A[4,7] * Conv_2_aggregated_features[7,12]
   + A[5,7] * Conv_2_aggregated_features[7,12]
   + A[6,7] * Conv_2_aggregated_features[7,12]
   - A[7,0] * Conv_1_Relu_ts[0,12] - A[7,1] * Conv_1_Relu_ts[1,12]
   - A[7,2] * Conv_1_Relu_ts[2,12] - A[7,3] * Conv_1_Relu_ts[3,12]
   - A[7,4] * Conv_1_Relu_ts[4,12] - A[7,5] * Conv_1_Relu_ts[5,12]
   - A[7,6] * Conv_1_Relu_ts[6,12] - A[7,7] * Conv_1_Relu_ts[7,12]
   + A[7,7] * Conv_2_aggregated_features[7,12] ] = 0
 Conv_2_averages_constraint[7,13]: [
   A[0,7] * Conv_2_aggregated_features[7,13]
   + A[1,7] * Conv_2_aggregated_features[7,13]
   + A[2,7] * Conv_2_aggregated_features[7,13]
   + A[3,7] * Conv_2_aggregated_features[7,13]
   + A[4,7] * Conv_2_aggregated_features[7,13]
   + A[5,7] * Conv_2_aggregated_features[7,13]
   + A[6,7] * Conv_2_aggregated_features[7,13]
   - A[7,0] * Conv_1_Relu_ts[0,13] - A[7,1] * Conv_1_Relu_ts[1,13]
   - A[7,2] * Conv_1_Relu_ts[2,13] - A[7,3] * Conv_1_Relu_ts[3,13]
   - A[7,4] * Conv_1_Relu_ts[4,13] - A[7,5] * Conv_1_Relu_ts[5,13]
   - A[7,6] * Conv_1_Relu_ts[6,13] - A[7,7] * Conv_1_Relu_ts[7,13]
   + A[7,7] * Conv_2_aggregated_features[7,13] ] = 0
 Conv_2_averages_constraint[7,14]: [
   A[0,7] * Conv_2_aggregated_features[7,14]
   + A[1,7] * Conv_2_aggregated_features[7,14]
   + A[2,7] * Conv_2_aggregated_features[7,14]
   + A[3,7] * Conv_2_aggregated_features[7,14]
   + A[4,7] * Conv_2_aggregated_features[7,14]
   + A[5,7] * Conv_2_aggregated_features[7,14]
   + A[6,7] * Conv_2_aggregated_features[7,14]
   - A[7,0] * Conv_1_Relu_ts[0,14] - A[7,1] * Conv_1_Relu_ts[1,14]
   - A[7,2] * Conv_1_Relu_ts[2,14] - A[7,3] * Conv_1_Relu_ts[3,14]
   - A[7,4] * Conv_1_Relu_ts[4,14] - A[7,5] * Conv_1_Relu_ts[5,14]
   - A[7,6] * Conv_1_Relu_ts[6,14] - A[7,7] * Conv_1_Relu_ts[7,14]
   + A[7,7] * Conv_2_aggregated_features[7,14] ] = 0
 Conv_2_averages_constraint[7,15]: [
   A[0,7] * Conv_2_aggregated_features[7,15]
   + A[1,7] * Conv_2_aggregated_features[7,15]
   + A[2,7] * Conv_2_aggregated_features[7,15]
   + A[3,7] * Conv_2_aggregated_features[7,15]
   + A[4,7] * Conv_2_aggregated_features[7,15]
   + A[5,7] * Conv_2_aggregated_features[7,15]
   + A[6,7] * Conv_2_aggregated_features[7,15]
   - A[7,0] * Conv_1_Relu_ts[0,15] - A[7,1] * Conv_1_Relu_ts[1,15]
   - A[7,2] * Conv_1_Relu_ts[2,15] - A[7,3] * Conv_1_Relu_ts[3,15]
   - A[7,4] * Conv_1_Relu_ts[4,15] - A[7,5] * Conv_1_Relu_ts[5,15]
   - A[7,6] * Conv_1_Relu_ts[6,15] - A[7,7] * Conv_1_Relu_ts[7,15]
   + A[7,7] * Conv_2_aggregated_features[7,15] ] = 0
 cosine_similarity: 7.735e-06 Aggregation[3] + 3.74893e-12 Aggregation[5]
   + 1.346948504447937 Aggregation[6] + 2.924052238464355 Aggregation[7]
   + 0.000227634 Aggregation[9] + 1.04378e-07 Aggregation[11]
   + 0.7807585000991821 Aggregation[12] + 0.000103446 Aggregation[13]
   + 4.538569450378418 Aggregation[14] + 1.73633e-06 Aggregation[15] + [
   - 5.618945598602295 embedding_similarity * embedding_magnitude ] = 0
Bounds
 X[0,0] <= 8
 X[1,0] <= 8
 X[2,0] <= 8
 X[3,0] <= 8
 X[4,0] <= 8
 X[5,0] <= 8
 X[6,0] <= 8
 X[7,0] <= 8
 Conv_0_aggregated_features[0,0] <= 8
 Conv_0_aggregated_features[1,0] <= 8
 Conv_0_aggregated_features[2,0] <= 8
 Conv_0_aggregated_features[3,0] <= 8
 Conv_0_aggregated_features[4,0] <= 8
 Conv_0_aggregated_features[5,0] <= 8
 Conv_0_aggregated_features[6,0] <= 8
 Conv_0_aggregated_features[7,0] <= 8
 -0.00345964082953287 <= Conv_0_t[0,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[0,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[0,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[0,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[0,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[0,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[0,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[0,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[0,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[0,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[0,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[0,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[0,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[0,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[0,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[0,15] <= -1.82729e-42
 -0.00345964082953287 <= Conv_0_t[1,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[1,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[1,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[1,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[1,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[1,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[1,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[1,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[1,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[1,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[1,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[1,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[1,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[1,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[1,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[1,15] <= -1.82729e-42
 -0.00345964082953287 <= Conv_0_t[2,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[2,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[2,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[2,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[2,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[2,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[2,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[2,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[2,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[2,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[2,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[2,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[2,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[2,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[2,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[2,15] <= -1.82729e-42
 -0.00345964082953287 <= Conv_0_t[3,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[3,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[3,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[3,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[3,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[3,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[3,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[3,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[3,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[3,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[3,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[3,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[3,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[3,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[3,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[3,15] <= -1.82729e-42
 -0.00345964082953287 <= Conv_0_t[4,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[4,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[4,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[4,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[4,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[4,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[4,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[4,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[4,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[4,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[4,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[4,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[4,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[4,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[4,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[4,15] <= -1.82729e-42
 -0.00345964082953287 <= Conv_0_t[5,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[5,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[5,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[5,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[5,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[5,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[5,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[5,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[5,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[5,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[5,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[5,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[5,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[5,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[5,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[5,15] <= -1.82729e-42
 -0.00345964082953287 <= Conv_0_t[6,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[6,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[6,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[6,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[6,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[6,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[6,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[6,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[6,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[6,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[6,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[6,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[6,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[6,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[6,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[6,15] <= -1.82729e-42
 -0.00345964082953287 <= Conv_0_t[7,0] <= -2.39668e-05
 -9.411394545342773 <= Conv_0_t[7,1] <= 9.188304475042969
 -3.183451279997826 <= Conv_0_t[7,2] <= 2.20557464659214
 -4.99524e-07 <= Conv_0_t[7,3] <= 1.22294e-14
 -0.00330359794190437 <= Conv_0_t[7,4] <= -1.44265e-10
 -0.00111407172748819 <= Conv_0_t[7,5] <= -2.319e-12
 -0.6694869505063252 <= Conv_0_t[7,6] <= -1.75046e-05
 -0.00719343813517304 <= Conv_0_t[7,7] <= -5.5964e-09
 -1.23637e-06 <= Conv_0_t[7,8] <= -5.52095e-08
 -3.06981e-09 <= Conv_0_t[7,9] <= -9.22955e-11
 -5.203926563262939 <= Conv_0_t[7,10] <= 3.856072187423706
 -0.00222280243337676 <= Conv_0_t[7,11] <= -1.58846e-12
 -1.310906410217285 <= Conv_0_t[7,12] <= 4.69843602180481
 -8.71338e-05 <= Conv_0_t[7,13] <= 8.63833e-06
 5.73562e-19 <= Conv_0_t[7,14] <= 8.39274e-18
 -2.94497e-41 <= Conv_0_t[7,15] <= -1.82729e-42
 Conv_0_Relu_ts[0,0] = 0
 Conv_0_Relu_ts[0,1] <= 9.188304475042969
 Conv_0_Relu_ts[0,2] <= 2.20557464659214
 Conv_0_Relu_ts[0,3] <= 1.22294e-14
 Conv_0_Relu_ts[0,4] = 0
 Conv_0_Relu_ts[0,5] = 0
 Conv_0_Relu_ts[0,6] = 0
 Conv_0_Relu_ts[0,7] = 0
 Conv_0_Relu_ts[0,8] = 0
 Conv_0_Relu_ts[0,9] = 0
 Conv_0_Relu_ts[0,10] <= 3.856072187423706
 Conv_0_Relu_ts[0,11] = 0
 Conv_0_Relu_ts[0,12] <= 4.69843602180481
 Conv_0_Relu_ts[0,13] <= 8.63833e-06
 Conv_0_Relu_ts[0,14] <= 8.39274e-18
 Conv_0_Relu_ts[0,15] = 0
 Conv_0_Relu_ts[1,0] = 0
 Conv_0_Relu_ts[1,1] <= 9.188304475042969
 Conv_0_Relu_ts[1,2] <= 2.20557464659214
 Conv_0_Relu_ts[1,3] <= 1.22294e-14
 Conv_0_Relu_ts[1,4] = 0
 Conv_0_Relu_ts[1,5] = 0
 Conv_0_Relu_ts[1,6] = 0
 Conv_0_Relu_ts[1,7] = 0
 Conv_0_Relu_ts[1,8] = 0
 Conv_0_Relu_ts[1,9] = 0
 Conv_0_Relu_ts[1,10] <= 3.856072187423706
 Conv_0_Relu_ts[1,11] = 0
 Conv_0_Relu_ts[1,12] <= 4.69843602180481
 Conv_0_Relu_ts[1,13] <= 8.63833e-06
 Conv_0_Relu_ts[1,14] <= 8.39274e-18
 Conv_0_Relu_ts[1,15] = 0
 Conv_0_Relu_ts[2,0] = 0
 Conv_0_Relu_ts[2,1] <= 9.188304475042969
 Conv_0_Relu_ts[2,2] <= 2.20557464659214
 Conv_0_Relu_ts[2,3] <= 1.22294e-14
 Conv_0_Relu_ts[2,4] = 0
 Conv_0_Relu_ts[2,5] = 0
 Conv_0_Relu_ts[2,6] = 0
 Conv_0_Relu_ts[2,7] = 0
 Conv_0_Relu_ts[2,8] = 0
 Conv_0_Relu_ts[2,9] = 0
 Conv_0_Relu_ts[2,10] <= 3.856072187423706
 Conv_0_Relu_ts[2,11] = 0
 Conv_0_Relu_ts[2,12] <= 4.69843602180481
 Conv_0_Relu_ts[2,13] <= 8.63833e-06
 Conv_0_Relu_ts[2,14] <= 8.39274e-18
 Conv_0_Relu_ts[2,15] = 0
 Conv_0_Relu_ts[3,0] = 0
 Conv_0_Relu_ts[3,1] <= 9.188304475042969
 Conv_0_Relu_ts[3,2] <= 2.20557464659214
 Conv_0_Relu_ts[3,3] <= 1.22294e-14
 Conv_0_Relu_ts[3,4] = 0
 Conv_0_Relu_ts[3,5] = 0
 Conv_0_Relu_ts[3,6] = 0
 Conv_0_Relu_ts[3,7] = 0
 Conv_0_Relu_ts[3,8] = 0
 Conv_0_Relu_ts[3,9] = 0
 Conv_0_Relu_ts[3,10] <= 3.856072187423706
 Conv_0_Relu_ts[3,11] = 0
 Conv_0_Relu_ts[3,12] <= 4.69843602180481
 Conv_0_Relu_ts[3,13] <= 8.63833e-06
 Conv_0_Relu_ts[3,14] <= 8.39274e-18
 Conv_0_Relu_ts[3,15] = 0
 Conv_0_Relu_ts[4,0] = 0
 Conv_0_Relu_ts[4,1] <= 9.188304475042969
 Conv_0_Relu_ts[4,2] <= 2.20557464659214
 Conv_0_Relu_ts[4,3] <= 1.22294e-14
 Conv_0_Relu_ts[4,4] = 0
 Conv_0_Relu_ts[4,5] = 0
 Conv_0_Relu_ts[4,6] = 0
 Conv_0_Relu_ts[4,7] = 0
 Conv_0_Relu_ts[4,8] = 0
 Conv_0_Relu_ts[4,9] = 0
 Conv_0_Relu_ts[4,10] <= 3.856072187423706
 Conv_0_Relu_ts[4,11] = 0
 Conv_0_Relu_ts[4,12] <= 4.69843602180481
 Conv_0_Relu_ts[4,13] <= 8.63833e-06
 Conv_0_Relu_ts[4,14] <= 8.39274e-18
 Conv_0_Relu_ts[4,15] = 0
 Conv_0_Relu_ts[5,0] = 0
 Conv_0_Relu_ts[5,1] <= 9.188304475042969
 Conv_0_Relu_ts[5,2] <= 2.20557464659214
 Conv_0_Relu_ts[5,3] <= 1.22294e-14
 Conv_0_Relu_ts[5,4] = 0
 Conv_0_Relu_ts[5,5] = 0
 Conv_0_Relu_ts[5,6] = 0
 Conv_0_Relu_ts[5,7] = 0
 Conv_0_Relu_ts[5,8] = 0
 Conv_0_Relu_ts[5,9] = 0
 Conv_0_Relu_ts[5,10] <= 3.856072187423706
 Conv_0_Relu_ts[5,11] = 0
 Conv_0_Relu_ts[5,12] <= 4.69843602180481
 Conv_0_Relu_ts[5,13] <= 8.63833e-06
 Conv_0_Relu_ts[5,14] <= 8.39274e-18
 Conv_0_Relu_ts[5,15] = 0
 Conv_0_Relu_ts[6,0] = 0
 Conv_0_Relu_ts[6,1] <= 9.188304475042969
 Conv_0_Relu_ts[6,2] <= 2.20557464659214
 Conv_0_Relu_ts[6,3] <= 1.22294e-14
 Conv_0_Relu_ts[6,4] = 0
 Conv_0_Relu_ts[6,5] = 0
 Conv_0_Relu_ts[6,6] = 0
 Conv_0_Relu_ts[6,7] = 0
 Conv_0_Relu_ts[6,8] = 0
 Conv_0_Relu_ts[6,9] = 0
 Conv_0_Relu_ts[6,10] <= 3.856072187423706
 Conv_0_Relu_ts[6,11] = 0
 Conv_0_Relu_ts[6,12] <= 4.69843602180481
 Conv_0_Relu_ts[6,13] <= 8.63833e-06
 Conv_0_Relu_ts[6,14] <= 8.39274e-18
 Conv_0_Relu_ts[6,15] = 0
 Conv_0_Relu_ts[7,0] = 0
 Conv_0_Relu_ts[7,1] <= 9.188304475042969
 Conv_0_Relu_ts[7,2] <= 2.20557464659214
 Conv_0_Relu_ts[7,3] <= 1.22294e-14
 Conv_0_Relu_ts[7,4] = 0
 Conv_0_Relu_ts[7,5] = 0
 Conv_0_Relu_ts[7,6] = 0
 Conv_0_Relu_ts[7,7] = 0
 Conv_0_Relu_ts[7,8] = 0
 Conv_0_Relu_ts[7,9] = 0
 Conv_0_Relu_ts[7,10] <= 3.856072187423706
 Conv_0_Relu_ts[7,11] = 0
 Conv_0_Relu_ts[7,12] <= 4.69843602180481
 Conv_0_Relu_ts[7,13] <= 8.63833e-06
 Conv_0_Relu_ts[7,14] <= 8.39274e-18
 Conv_0_Relu_ts[7,15] = 0
 Conv_0_Relu_ss[0,0] <= 0.00345964082953287
 Conv_0_Relu_ss[0,1] <= 9.411394545342773
 Conv_0_Relu_ss[0,2] <= 3.183451279997826
 Conv_0_Relu_ss[0,3] <= 4.99524e-07
 Conv_0_Relu_ss[0,4] <= 0.00330359794190437
 Conv_0_Relu_ss[0,5] <= 0.00111407172748819
 Conv_0_Relu_ss[0,6] <= 0.6694869505063252
 Conv_0_Relu_ss[0,7] <= 0.00719343813517304
 Conv_0_Relu_ss[0,8] <= 1.23637e-06
 Conv_0_Relu_ss[0,9] <= 3.06981e-09
 Conv_0_Relu_ss[0,10] <= 5.203926563262939
 Conv_0_Relu_ss[0,11] <= 0.00222280243337676
 Conv_0_Relu_ss[0,12] <= 1.310906410217285
 Conv_0_Relu_ss[0,13] <= 8.71338e-05
 Conv_0_Relu_ss[0,14] = 0
 Conv_0_Relu_ss[0,15] <= 2.94497e-41
 Conv_0_Relu_ss[1,0] <= 0.00345964082953287
 Conv_0_Relu_ss[1,1] <= 9.411394545342773
 Conv_0_Relu_ss[1,2] <= 3.183451279997826
 Conv_0_Relu_ss[1,3] <= 4.99524e-07
 Conv_0_Relu_ss[1,4] <= 0.00330359794190437
 Conv_0_Relu_ss[1,5] <= 0.00111407172748819
 Conv_0_Relu_ss[1,6] <= 0.6694869505063252
 Conv_0_Relu_ss[1,7] <= 0.00719343813517304
 Conv_0_Relu_ss[1,8] <= 1.23637e-06
 Conv_0_Relu_ss[1,9] <= 3.06981e-09
 Conv_0_Relu_ss[1,10] <= 5.203926563262939
 Conv_0_Relu_ss[1,11] <= 0.00222280243337676
 Conv_0_Relu_ss[1,12] <= 1.310906410217285
 Conv_0_Relu_ss[1,13] <= 8.71338e-05
 Conv_0_Relu_ss[1,14] = 0
 Conv_0_Relu_ss[1,15] <= 2.94497e-41
 Conv_0_Relu_ss[2,0] <= 0.00345964082953287
 Conv_0_Relu_ss[2,1] <= 9.411394545342773
 Conv_0_Relu_ss[2,2] <= 3.183451279997826
 Conv_0_Relu_ss[2,3] <= 4.99524e-07
 Conv_0_Relu_ss[2,4] <= 0.00330359794190437
 Conv_0_Relu_ss[2,5] <= 0.00111407172748819
 Conv_0_Relu_ss[2,6] <= 0.6694869505063252
 Conv_0_Relu_ss[2,7] <= 0.00719343813517304
 Conv_0_Relu_ss[2,8] <= 1.23637e-06
 Conv_0_Relu_ss[2,9] <= 3.06981e-09
 Conv_0_Relu_ss[2,10] <= 5.203926563262939
 Conv_0_Relu_ss[2,11] <= 0.00222280243337676
 Conv_0_Relu_ss[2,12] <= 1.310906410217285
 Conv_0_Relu_ss[2,13] <= 8.71338e-05
 Conv_0_Relu_ss[2,14] = 0
 Conv_0_Relu_ss[2,15] <= 2.94497e-41
 Conv_0_Relu_ss[3,0] <= 0.00345964082953287
 Conv_0_Relu_ss[3,1] <= 9.411394545342773
 Conv_0_Relu_ss[3,2] <= 3.183451279997826
 Conv_0_Relu_ss[3,3] <= 4.99524e-07
 Conv_0_Relu_ss[3,4] <= 0.00330359794190437
 Conv_0_Relu_ss[3,5] <= 0.00111407172748819
 Conv_0_Relu_ss[3,6] <= 0.6694869505063252
 Conv_0_Relu_ss[3,7] <= 0.00719343813517304
 Conv_0_Relu_ss[3,8] <= 1.23637e-06
 Conv_0_Relu_ss[3,9] <= 3.06981e-09
 Conv_0_Relu_ss[3,10] <= 5.203926563262939
 Conv_0_Relu_ss[3,11] <= 0.00222280243337676
 Conv_0_Relu_ss[3,12] <= 1.310906410217285
 Conv_0_Relu_ss[3,13] <= 8.71338e-05
 Conv_0_Relu_ss[3,14] = 0
 Conv_0_Relu_ss[3,15] <= 2.94497e-41
 Conv_0_Relu_ss[4,0] <= 0.00345964082953287
 Conv_0_Relu_ss[4,1] <= 9.411394545342773
 Conv_0_Relu_ss[4,2] <= 3.183451279997826
 Conv_0_Relu_ss[4,3] <= 4.99524e-07
 Conv_0_Relu_ss[4,4] <= 0.00330359794190437
 Conv_0_Relu_ss[4,5] <= 0.00111407172748819
 Conv_0_Relu_ss[4,6] <= 0.6694869505063252
 Conv_0_Relu_ss[4,7] <= 0.00719343813517304
 Conv_0_Relu_ss[4,8] <= 1.23637e-06
 Conv_0_Relu_ss[4,9] <= 3.06981e-09
 Conv_0_Relu_ss[4,10] <= 5.203926563262939
 Conv_0_Relu_ss[4,11] <= 0.00222280243337676
 Conv_0_Relu_ss[4,12] <= 1.310906410217285
 Conv_0_Relu_ss[4,13] <= 8.71338e-05
 Conv_0_Relu_ss[4,14] = 0
 Conv_0_Relu_ss[4,15] <= 2.94497e-41
 Conv_0_Relu_ss[5,0] <= 0.00345964082953287
 Conv_0_Relu_ss[5,1] <= 9.411394545342773
 Conv_0_Relu_ss[5,2] <= 3.183451279997826
 Conv_0_Relu_ss[5,3] <= 4.99524e-07
 Conv_0_Relu_ss[5,4] <= 0.00330359794190437
 Conv_0_Relu_ss[5,5] <= 0.00111407172748819
 Conv_0_Relu_ss[5,6] <= 0.6694869505063252
 Conv_0_Relu_ss[5,7] <= 0.00719343813517304
 Conv_0_Relu_ss[5,8] <= 1.23637e-06
 Conv_0_Relu_ss[5,9] <= 3.06981e-09
 Conv_0_Relu_ss[5,10] <= 5.203926563262939
 Conv_0_Relu_ss[5,11] <= 0.00222280243337676
 Conv_0_Relu_ss[5,12] <= 1.310906410217285
 Conv_0_Relu_ss[5,13] <= 8.71338e-05
 Conv_0_Relu_ss[5,14] = 0
 Conv_0_Relu_ss[5,15] <= 2.94497e-41
 Conv_0_Relu_ss[6,0] <= 0.00345964082953287
 Conv_0_Relu_ss[6,1] <= 9.411394545342773
 Conv_0_Relu_ss[6,2] <= 3.183451279997826
 Conv_0_Relu_ss[6,3] <= 4.99524e-07
 Conv_0_Relu_ss[6,4] <= 0.00330359794190437
 Conv_0_Relu_ss[6,5] <= 0.00111407172748819
 Conv_0_Relu_ss[6,6] <= 0.6694869505063252
 Conv_0_Relu_ss[6,7] <= 0.00719343813517304
 Conv_0_Relu_ss[6,8] <= 1.23637e-06
 Conv_0_Relu_ss[6,9] <= 3.06981e-09
 Conv_0_Relu_ss[6,10] <= 5.203926563262939
 Conv_0_Relu_ss[6,11] <= 0.00222280243337676
 Conv_0_Relu_ss[6,12] <= 1.310906410217285
 Conv_0_Relu_ss[6,13] <= 8.71338e-05
 Conv_0_Relu_ss[6,14] = 0
 Conv_0_Relu_ss[6,15] <= 2.94497e-41
 Conv_0_Relu_ss[7,0] <= 0.00345964082953287
 Conv_0_Relu_ss[7,1] <= 9.411394545342773
 Conv_0_Relu_ss[7,2] <= 3.183451279997826
 Conv_0_Relu_ss[7,3] <= 4.99524e-07
 Conv_0_Relu_ss[7,4] <= 0.00330359794190437
 Conv_0_Relu_ss[7,5] <= 0.00111407172748819
 Conv_0_Relu_ss[7,6] <= 0.6694869505063252
 Conv_0_Relu_ss[7,7] <= 0.00719343813517304
 Conv_0_Relu_ss[7,8] <= 1.23637e-06
 Conv_0_Relu_ss[7,9] <= 3.06981e-09
 Conv_0_Relu_ss[7,10] <= 5.203926563262939
 Conv_0_Relu_ss[7,11] <= 0.00222280243337676
 Conv_0_Relu_ss[7,12] <= 1.310906410217285
 Conv_0_Relu_ss[7,13] <= 8.71338e-05
 Conv_0_Relu_ss[7,14] = 0
 Conv_0_Relu_ss[7,15] <= 2.94497e-41
 Conv_1_aggregated_features[0,0] = 0
 Conv_1_aggregated_features[0,1] <= 9.188304475042969
 Conv_1_aggregated_features[0,2] <= 2.20557464659214
 Conv_1_aggregated_features[0,3] <= 1.22294e-14
 Conv_1_aggregated_features[0,4] = 0
 Conv_1_aggregated_features[0,5] = 0
 Conv_1_aggregated_features[0,6] = 0
 Conv_1_aggregated_features[0,7] = 0
 Conv_1_aggregated_features[0,8] = 0
 Conv_1_aggregated_features[0,9] = 0
 Conv_1_aggregated_features[0,10] <= 3.856072187423706
 Conv_1_aggregated_features[0,11] = 0
 Conv_1_aggregated_features[0,12] <= 4.69843602180481
 Conv_1_aggregated_features[0,13] <= 8.63833e-06
 Conv_1_aggregated_features[0,14] <= 8.39274e-18
 Conv_1_aggregated_features[0,15] = 0
 Conv_1_aggregated_features[1,0] = 0
 Conv_1_aggregated_features[1,1] <= 9.188304475042969
 Conv_1_aggregated_features[1,2] <= 2.20557464659214
 Conv_1_aggregated_features[1,3] <= 1.22294e-14
 Conv_1_aggregated_features[1,4] = 0
 Conv_1_aggregated_features[1,5] = 0
 Conv_1_aggregated_features[1,6] = 0
 Conv_1_aggregated_features[1,7] = 0
 Conv_1_aggregated_features[1,8] = 0
 Conv_1_aggregated_features[1,9] = 0
 Conv_1_aggregated_features[1,10] <= 3.856072187423706
 Conv_1_aggregated_features[1,11] = 0
 Conv_1_aggregated_features[1,12] <= 4.69843602180481
 Conv_1_aggregated_features[1,13] <= 8.63833e-06
 Conv_1_aggregated_features[1,14] <= 8.39274e-18
 Conv_1_aggregated_features[1,15] = 0
 Conv_1_aggregated_features[2,0] = 0
 Conv_1_aggregated_features[2,1] <= 9.188304475042969
 Conv_1_aggregated_features[2,2] <= 2.20557464659214
 Conv_1_aggregated_features[2,3] <= 1.22294e-14
 Conv_1_aggregated_features[2,4] = 0
 Conv_1_aggregated_features[2,5] = 0
 Conv_1_aggregated_features[2,6] = 0
 Conv_1_aggregated_features[2,7] = 0
 Conv_1_aggregated_features[2,8] = 0
 Conv_1_aggregated_features[2,9] = 0
 Conv_1_aggregated_features[2,10] <= 3.856072187423706
 Conv_1_aggregated_features[2,11] = 0
 Conv_1_aggregated_features[2,12] <= 4.69843602180481
 Conv_1_aggregated_features[2,13] <= 8.63833e-06
 Conv_1_aggregated_features[2,14] <= 8.39274e-18
 Conv_1_aggregated_features[2,15] = 0
 Conv_1_aggregated_features[3,0] = 0
 Conv_1_aggregated_features[3,1] <= 9.188304475042969
 Conv_1_aggregated_features[3,2] <= 2.20557464659214
 Conv_1_aggregated_features[3,3] <= 1.22294e-14
 Conv_1_aggregated_features[3,4] = 0
 Conv_1_aggregated_features[3,5] = 0
 Conv_1_aggregated_features[3,6] = 0
 Conv_1_aggregated_features[3,7] = 0
 Conv_1_aggregated_features[3,8] = 0
 Conv_1_aggregated_features[3,9] = 0
 Conv_1_aggregated_features[3,10] <= 3.856072187423706
 Conv_1_aggregated_features[3,11] = 0
 Conv_1_aggregated_features[3,12] <= 4.69843602180481
 Conv_1_aggregated_features[3,13] <= 8.63833e-06
 Conv_1_aggregated_features[3,14] <= 8.39274e-18
 Conv_1_aggregated_features[3,15] = 0
 Conv_1_aggregated_features[4,0] = 0
 Conv_1_aggregated_features[4,1] <= 9.188304475042969
 Conv_1_aggregated_features[4,2] <= 2.20557464659214
 Conv_1_aggregated_features[4,3] <= 1.22294e-14
 Conv_1_aggregated_features[4,4] = 0
 Conv_1_aggregated_features[4,5] = 0
 Conv_1_aggregated_features[4,6] = 0
 Conv_1_aggregated_features[4,7] = 0
 Conv_1_aggregated_features[4,8] = 0
 Conv_1_aggregated_features[4,9] = 0
 Conv_1_aggregated_features[4,10] <= 3.856072187423706
 Conv_1_aggregated_features[4,11] = 0
 Conv_1_aggregated_features[4,12] <= 4.69843602180481
 Conv_1_aggregated_features[4,13] <= 8.63833e-06
 Conv_1_aggregated_features[4,14] <= 8.39274e-18
 Conv_1_aggregated_features[4,15] = 0
 Conv_1_aggregated_features[5,0] = 0
 Conv_1_aggregated_features[5,1] <= 9.188304475042969
 Conv_1_aggregated_features[5,2] <= 2.20557464659214
 Conv_1_aggregated_features[5,3] <= 1.22294e-14
 Conv_1_aggregated_features[5,4] = 0
 Conv_1_aggregated_features[5,5] = 0
 Conv_1_aggregated_features[5,6] = 0
 Conv_1_aggregated_features[5,7] = 0
 Conv_1_aggregated_features[5,8] = 0
 Conv_1_aggregated_features[5,9] = 0
 Conv_1_aggregated_features[5,10] <= 3.856072187423706
 Conv_1_aggregated_features[5,11] = 0
 Conv_1_aggregated_features[5,12] <= 4.69843602180481
 Conv_1_aggregated_features[5,13] <= 8.63833e-06
 Conv_1_aggregated_features[5,14] <= 8.39274e-18
 Conv_1_aggregated_features[5,15] = 0
 Conv_1_aggregated_features[6,0] = 0
 Conv_1_aggregated_features[6,1] <= 9.188304475042969
 Conv_1_aggregated_features[6,2] <= 2.20557464659214
 Conv_1_aggregated_features[6,3] <= 1.22294e-14
 Conv_1_aggregated_features[6,4] = 0
 Conv_1_aggregated_features[6,5] = 0
 Conv_1_aggregated_features[6,6] = 0
 Conv_1_aggregated_features[6,7] = 0
 Conv_1_aggregated_features[6,8] = 0
 Conv_1_aggregated_features[6,9] = 0
 Conv_1_aggregated_features[6,10] <= 3.856072187423706
 Conv_1_aggregated_features[6,11] = 0
 Conv_1_aggregated_features[6,12] <= 4.69843602180481
 Conv_1_aggregated_features[6,13] <= 8.63833e-06
 Conv_1_aggregated_features[6,14] <= 8.39274e-18
 Conv_1_aggregated_features[6,15] = 0
 Conv_1_aggregated_features[7,0] = 0
 Conv_1_aggregated_features[7,1] <= 9.188304475042969
 Conv_1_aggregated_features[7,2] <= 2.20557464659214
 Conv_1_aggregated_features[7,3] <= 1.22294e-14
 Conv_1_aggregated_features[7,4] = 0
 Conv_1_aggregated_features[7,5] = 0
 Conv_1_aggregated_features[7,6] = 0
 Conv_1_aggregated_features[7,7] = 0
 Conv_1_aggregated_features[7,8] = 0
 Conv_1_aggregated_features[7,9] = 0
 Conv_1_aggregated_features[7,10] <= 3.856072187423706
 Conv_1_aggregated_features[7,11] = 0
 Conv_1_aggregated_features[7,12] <= 4.69843602180481
 Conv_1_aggregated_features[7,13] <= 8.63833e-06
 Conv_1_aggregated_features[7,14] <= 8.39274e-18
 Conv_1_aggregated_features[7,15] = 0
 -0.00554961914110762 <= Conv_1_t[0,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[0,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[0,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[0,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[0,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[0,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[0,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[0,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[0,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[0,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[0,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[0,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[0,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[0,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[0,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[0,15] <= -1.03481e-08
 -0.00554961914110762 <= Conv_1_t[1,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[1,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[1,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[1,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[1,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[1,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[1,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[1,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[1,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[1,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[1,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[1,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[1,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[1,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[1,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[1,15] <= -1.03481e-08
 -0.00554961914110762 <= Conv_1_t[2,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[2,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[2,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[2,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[2,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[2,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[2,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[2,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[2,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[2,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[2,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[2,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[2,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[2,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[2,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[2,15] <= -1.03481e-08
 -0.00554961914110762 <= Conv_1_t[3,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[3,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[3,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[3,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[3,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[3,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[3,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[3,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[3,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[3,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[3,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[3,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[3,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[3,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[3,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[3,15] <= -1.03481e-08
 -0.00554961914110762 <= Conv_1_t[4,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[4,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[4,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[4,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[4,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[4,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[4,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[4,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[4,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[4,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[4,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[4,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[4,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[4,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[4,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[4,15] <= -1.03481e-08
 -0.00554961914110762 <= Conv_1_t[5,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[5,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[5,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[5,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[5,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[5,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[5,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[5,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[5,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[5,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[5,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[5,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[5,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[5,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[5,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[5,15] <= -1.03481e-08
 -0.00554961914110762 <= Conv_1_t[6,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[6,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[6,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[6,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[6,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[6,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[6,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[6,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[6,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[6,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[6,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[6,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[6,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[6,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[6,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[6,15] <= -1.03481e-08
 -0.00554961914110762 <= Conv_1_t[7,0] <= -1.7536e-08
 -1.857019803605887 <= Conv_1_t[7,1] <= 14.59003230898199
 -1.05164e-07 <= Conv_1_t[7,2] <= -1.53413e-08
 -0.00336194144980918 <= Conv_1_t[7,3] <= -3.15407e-06
 -3.66243e-06 <= Conv_1_t[7,4] <= -8.55466e-08
 -2.93491e-08 <= Conv_1_t[7,5] <= -2.61297e-09
 -5.8459259755839165e-04 <= Conv_1_t[7,6] <= -7.76048e-05
 -1.08295e-07 <= Conv_1_t[7,7] <= -3.26654e-09
 -1.27821e-12 <= Conv_1_t[7,8] <= 2.75325e-13
 -4.958788286540738 <= Conv_1_t[7,9] <= 8.234055808249286
 -1.45772e-05 <= Conv_1_t[7,10] <= -6.26579e-08
 -0.00236660145696522 <= Conv_1_t[7,11] <= 3.07161e-05
 -3.08689e-07 <= Conv_1_t[7,12] <= -1.09074e-09
 -4.431268808214668 <= Conv_1_t[7,13] <= 11.92005955570159
 -0.178791987482969 <= Conv_1_t[7,14] <= 0.00295504399667217
 -1.22225e-07 <= Conv_1_t[7,15] <= -1.03481e-08
 Conv_1_Relu_ts[0,0] = 0
 Conv_1_Relu_ts[0,1] <= 14.59003230898199
 Conv_1_Relu_ts[0,2] = 0
 Conv_1_Relu_ts[0,3] = 0
 Conv_1_Relu_ts[0,4] = 0
 Conv_1_Relu_ts[0,5] = 0
 Conv_1_Relu_ts[0,6] = 0
 Conv_1_Relu_ts[0,7] = 0
 Conv_1_Relu_ts[0,8] <= 2.75325e-13
 Conv_1_Relu_ts[0,9] <= 8.234055808249286
 Conv_1_Relu_ts[0,10] = 0
 Conv_1_Relu_ts[0,11] <= 3.07161e-05
 Conv_1_Relu_ts[0,12] = 0
 Conv_1_Relu_ts[0,13] <= 11.92005955570159
 Conv_1_Relu_ts[0,14] <= 0.00295504399667217
 Conv_1_Relu_ts[0,15] = 0
 Conv_1_Relu_ts[1,0] = 0
 Conv_1_Relu_ts[1,1] <= 14.59003230898199
 Conv_1_Relu_ts[1,2] = 0
 Conv_1_Relu_ts[1,3] = 0
 Conv_1_Relu_ts[1,4] = 0
 Conv_1_Relu_ts[1,5] = 0
 Conv_1_Relu_ts[1,6] = 0
 Conv_1_Relu_ts[1,7] = 0
 Conv_1_Relu_ts[1,8] <= 2.75325e-13
 Conv_1_Relu_ts[1,9] <= 8.234055808249286
 Conv_1_Relu_ts[1,10] = 0
 Conv_1_Relu_ts[1,11] <= 3.07161e-05
 Conv_1_Relu_ts[1,12] = 0
 Conv_1_Relu_ts[1,13] <= 11.92005955570159
 Conv_1_Relu_ts[1,14] <= 0.00295504399667217
 Conv_1_Relu_ts[1,15] = 0
 Conv_1_Relu_ts[2,0] = 0
 Conv_1_Relu_ts[2,1] <= 14.59003230898199
 Conv_1_Relu_ts[2,2] = 0
 Conv_1_Relu_ts[2,3] = 0
 Conv_1_Relu_ts[2,4] = 0
 Conv_1_Relu_ts[2,5] = 0
 Conv_1_Relu_ts[2,6] = 0
 Conv_1_Relu_ts[2,7] = 0
 Conv_1_Relu_ts[2,8] <= 2.75325e-13
 Conv_1_Relu_ts[2,9] <= 8.234055808249286
 Conv_1_Relu_ts[2,10] = 0
 Conv_1_Relu_ts[2,11] <= 3.07161e-05
 Conv_1_Relu_ts[2,12] = 0
 Conv_1_Relu_ts[2,13] <= 11.92005955570159
 Conv_1_Relu_ts[2,14] <= 0.00295504399667217
 Conv_1_Relu_ts[2,15] = 0
 Conv_1_Relu_ts[3,0] = 0
 Conv_1_Relu_ts[3,1] <= 14.59003230898199
 Conv_1_Relu_ts[3,2] = 0
 Conv_1_Relu_ts[3,3] = 0
 Conv_1_Relu_ts[3,4] = 0
 Conv_1_Relu_ts[3,5] = 0
 Conv_1_Relu_ts[3,6] = 0
 Conv_1_Relu_ts[3,7] = 0
 Conv_1_Relu_ts[3,8] <= 2.75325e-13
 Conv_1_Relu_ts[3,9] <= 8.234055808249286
 Conv_1_Relu_ts[3,10] = 0
 Conv_1_Relu_ts[3,11] <= 3.07161e-05
 Conv_1_Relu_ts[3,12] = 0
 Conv_1_Relu_ts[3,13] <= 11.92005955570159
 Conv_1_Relu_ts[3,14] <= 0.00295504399667217
 Conv_1_Relu_ts[3,15] = 0
 Conv_1_Relu_ts[4,0] = 0
 Conv_1_Relu_ts[4,1] <= 14.59003230898199
 Conv_1_Relu_ts[4,2] = 0
 Conv_1_Relu_ts[4,3] = 0
 Conv_1_Relu_ts[4,4] = 0
 Conv_1_Relu_ts[4,5] = 0
 Conv_1_Relu_ts[4,6] = 0
 Conv_1_Relu_ts[4,7] = 0
 Conv_1_Relu_ts[4,8] <= 2.75325e-13
 Conv_1_Relu_ts[4,9] <= 8.234055808249286
 Conv_1_Relu_ts[4,10] = 0
 Conv_1_Relu_ts[4,11] <= 3.07161e-05
 Conv_1_Relu_ts[4,12] = 0
 Conv_1_Relu_ts[4,13] <= 11.92005955570159
 Conv_1_Relu_ts[4,14] <= 0.00295504399667217
 Conv_1_Relu_ts[4,15] = 0
 Conv_1_Relu_ts[5,0] = 0
 Conv_1_Relu_ts[5,1] <= 14.59003230898199
 Conv_1_Relu_ts[5,2] = 0
 Conv_1_Relu_ts[5,3] = 0
 Conv_1_Relu_ts[5,4] = 0
 Conv_1_Relu_ts[5,5] = 0
 Conv_1_Relu_ts[5,6] = 0
 Conv_1_Relu_ts[5,7] = 0
 Conv_1_Relu_ts[5,8] <= 2.75325e-13
 Conv_1_Relu_ts[5,9] <= 8.234055808249286
 Conv_1_Relu_ts[5,10] = 0
 Conv_1_Relu_ts[5,11] <= 3.07161e-05
 Conv_1_Relu_ts[5,12] = 0
 Conv_1_Relu_ts[5,13] <= 11.92005955570159
 Conv_1_Relu_ts[5,14] <= 0.00295504399667217
 Conv_1_Relu_ts[5,15] = 0
 Conv_1_Relu_ts[6,0] = 0
 Conv_1_Relu_ts[6,1] <= 14.59003230898199
 Conv_1_Relu_ts[6,2] = 0
 Conv_1_Relu_ts[6,3] = 0
 Conv_1_Relu_ts[6,4] = 0
 Conv_1_Relu_ts[6,5] = 0
 Conv_1_Relu_ts[6,6] = 0
 Conv_1_Relu_ts[6,7] = 0
 Conv_1_Relu_ts[6,8] <= 2.75325e-13
 Conv_1_Relu_ts[6,9] <= 8.234055808249286
 Conv_1_Relu_ts[6,10] = 0
 Conv_1_Relu_ts[6,11] <= 3.07161e-05
 Conv_1_Relu_ts[6,12] = 0
 Conv_1_Relu_ts[6,13] <= 11.92005955570159
 Conv_1_Relu_ts[6,14] <= 0.00295504399667217
 Conv_1_Relu_ts[6,15] = 0
 Conv_1_Relu_ts[7,0] = 0
 Conv_1_Relu_ts[7,1] <= 14.59003230898199
 Conv_1_Relu_ts[7,2] = 0
 Conv_1_Relu_ts[7,3] = 0
 Conv_1_Relu_ts[7,4] = 0
 Conv_1_Relu_ts[7,5] = 0
 Conv_1_Relu_ts[7,6] = 0
 Conv_1_Relu_ts[7,7] = 0
 Conv_1_Relu_ts[7,8] <= 2.75325e-13
 Conv_1_Relu_ts[7,9] <= 8.234055808249286
 Conv_1_Relu_ts[7,10] = 0
 Conv_1_Relu_ts[7,11] <= 3.07161e-05
 Conv_1_Relu_ts[7,12] = 0
 Conv_1_Relu_ts[7,13] <= 11.92005955570159
 Conv_1_Relu_ts[7,14] <= 0.00295504399667217
 Conv_1_Relu_ts[7,15] = 0
 Conv_1_Relu_ss[0,0] <= 0.00554961914110762
 Conv_1_Relu_ss[0,1] <= 1.857019803605887
 Conv_1_Relu_ss[0,2] <= 1.05164e-07
 Conv_1_Relu_ss[0,3] <= 0.00336194144980918
 Conv_1_Relu_ss[0,4] <= 3.66243e-06
 Conv_1_Relu_ss[0,5] <= 2.93491e-08
 Conv_1_Relu_ss[0,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[0,7] <= 1.08295e-07
 Conv_1_Relu_ss[0,8] <= 1.27821e-12
 Conv_1_Relu_ss[0,9] <= 4.958788286540738
 Conv_1_Relu_ss[0,10] <= 1.45772e-05
 Conv_1_Relu_ss[0,11] <= 0.00236660145696522
 Conv_1_Relu_ss[0,12] <= 3.08689e-07
 Conv_1_Relu_ss[0,13] <= 4.431268808214668
 Conv_1_Relu_ss[0,14] <= 0.178791987482969
 Conv_1_Relu_ss[0,15] <= 1.22225e-07
 Conv_1_Relu_ss[1,0] <= 0.00554961914110762
 Conv_1_Relu_ss[1,1] <= 1.857019803605887
 Conv_1_Relu_ss[1,2] <= 1.05164e-07
 Conv_1_Relu_ss[1,3] <= 0.00336194144980918
 Conv_1_Relu_ss[1,4] <= 3.66243e-06
 Conv_1_Relu_ss[1,5] <= 2.93491e-08
 Conv_1_Relu_ss[1,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[1,7] <= 1.08295e-07
 Conv_1_Relu_ss[1,8] <= 1.27821e-12
 Conv_1_Relu_ss[1,9] <= 4.958788286540738
 Conv_1_Relu_ss[1,10] <= 1.45772e-05
 Conv_1_Relu_ss[1,11] <= 0.00236660145696522
 Conv_1_Relu_ss[1,12] <= 3.08689e-07
 Conv_1_Relu_ss[1,13] <= 4.431268808214668
 Conv_1_Relu_ss[1,14] <= 0.178791987482969
 Conv_1_Relu_ss[1,15] <= 1.22225e-07
 Conv_1_Relu_ss[2,0] <= 0.00554961914110762
 Conv_1_Relu_ss[2,1] <= 1.857019803605887
 Conv_1_Relu_ss[2,2] <= 1.05164e-07
 Conv_1_Relu_ss[2,3] <= 0.00336194144980918
 Conv_1_Relu_ss[2,4] <= 3.66243e-06
 Conv_1_Relu_ss[2,5] <= 2.93491e-08
 Conv_1_Relu_ss[2,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[2,7] <= 1.08295e-07
 Conv_1_Relu_ss[2,8] <= 1.27821e-12
 Conv_1_Relu_ss[2,9] <= 4.958788286540738
 Conv_1_Relu_ss[2,10] <= 1.45772e-05
 Conv_1_Relu_ss[2,11] <= 0.00236660145696522
 Conv_1_Relu_ss[2,12] <= 3.08689e-07
 Conv_1_Relu_ss[2,13] <= 4.431268808214668
 Conv_1_Relu_ss[2,14] <= 0.178791987482969
 Conv_1_Relu_ss[2,15] <= 1.22225e-07
 Conv_1_Relu_ss[3,0] <= 0.00554961914110762
 Conv_1_Relu_ss[3,1] <= 1.857019803605887
 Conv_1_Relu_ss[3,2] <= 1.05164e-07
 Conv_1_Relu_ss[3,3] <= 0.00336194144980918
 Conv_1_Relu_ss[3,4] <= 3.66243e-06
 Conv_1_Relu_ss[3,5] <= 2.93491e-08
 Conv_1_Relu_ss[3,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[3,7] <= 1.08295e-07
 Conv_1_Relu_ss[3,8] <= 1.27821e-12
 Conv_1_Relu_ss[3,9] <= 4.958788286540738
 Conv_1_Relu_ss[3,10] <= 1.45772e-05
 Conv_1_Relu_ss[3,11] <= 0.00236660145696522
 Conv_1_Relu_ss[3,12] <= 3.08689e-07
 Conv_1_Relu_ss[3,13] <= 4.431268808214668
 Conv_1_Relu_ss[3,14] <= 0.178791987482969
 Conv_1_Relu_ss[3,15] <= 1.22225e-07
 Conv_1_Relu_ss[4,0] <= 0.00554961914110762
 Conv_1_Relu_ss[4,1] <= 1.857019803605887
 Conv_1_Relu_ss[4,2] <= 1.05164e-07
 Conv_1_Relu_ss[4,3] <= 0.00336194144980918
 Conv_1_Relu_ss[4,4] <= 3.66243e-06
 Conv_1_Relu_ss[4,5] <= 2.93491e-08
 Conv_1_Relu_ss[4,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[4,7] <= 1.08295e-07
 Conv_1_Relu_ss[4,8] <= 1.27821e-12
 Conv_1_Relu_ss[4,9] <= 4.958788286540738
 Conv_1_Relu_ss[4,10] <= 1.45772e-05
 Conv_1_Relu_ss[4,11] <= 0.00236660145696522
 Conv_1_Relu_ss[4,12] <= 3.08689e-07
 Conv_1_Relu_ss[4,13] <= 4.431268808214668
 Conv_1_Relu_ss[4,14] <= 0.178791987482969
 Conv_1_Relu_ss[4,15] <= 1.22225e-07
 Conv_1_Relu_ss[5,0] <= 0.00554961914110762
 Conv_1_Relu_ss[5,1] <= 1.857019803605887
 Conv_1_Relu_ss[5,2] <= 1.05164e-07
 Conv_1_Relu_ss[5,3] <= 0.00336194144980918
 Conv_1_Relu_ss[5,4] <= 3.66243e-06
 Conv_1_Relu_ss[5,5] <= 2.93491e-08
 Conv_1_Relu_ss[5,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[5,7] <= 1.08295e-07
 Conv_1_Relu_ss[5,8] <= 1.27821e-12
 Conv_1_Relu_ss[5,9] <= 4.958788286540738
 Conv_1_Relu_ss[5,10] <= 1.45772e-05
 Conv_1_Relu_ss[5,11] <= 0.00236660145696522
 Conv_1_Relu_ss[5,12] <= 3.08689e-07
 Conv_1_Relu_ss[5,13] <= 4.431268808214668
 Conv_1_Relu_ss[5,14] <= 0.178791987482969
 Conv_1_Relu_ss[5,15] <= 1.22225e-07
 Conv_1_Relu_ss[6,0] <= 0.00554961914110762
 Conv_1_Relu_ss[6,1] <= 1.857019803605887
 Conv_1_Relu_ss[6,2] <= 1.05164e-07
 Conv_1_Relu_ss[6,3] <= 0.00336194144980918
 Conv_1_Relu_ss[6,4] <= 3.66243e-06
 Conv_1_Relu_ss[6,5] <= 2.93491e-08
 Conv_1_Relu_ss[6,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[6,7] <= 1.08295e-07
 Conv_1_Relu_ss[6,8] <= 1.27821e-12
 Conv_1_Relu_ss[6,9] <= 4.958788286540738
 Conv_1_Relu_ss[6,10] <= 1.45772e-05
 Conv_1_Relu_ss[6,11] <= 0.00236660145696522
 Conv_1_Relu_ss[6,12] <= 3.08689e-07
 Conv_1_Relu_ss[6,13] <= 4.431268808214668
 Conv_1_Relu_ss[6,14] <= 0.178791987482969
 Conv_1_Relu_ss[6,15] <= 1.22225e-07
 Conv_1_Relu_ss[7,0] <= 0.00554961914110762
 Conv_1_Relu_ss[7,1] <= 1.857019803605887
 Conv_1_Relu_ss[7,2] <= 1.05164e-07
 Conv_1_Relu_ss[7,3] <= 0.00336194144980918
 Conv_1_Relu_ss[7,4] <= 3.66243e-06
 Conv_1_Relu_ss[7,5] <= 2.93491e-08
 Conv_1_Relu_ss[7,6] <= 5.8459259755839165e-04
 Conv_1_Relu_ss[7,7] <= 1.08295e-07
 Conv_1_Relu_ss[7,8] <= 1.27821e-12
 Conv_1_Relu_ss[7,9] <= 4.958788286540738
 Conv_1_Relu_ss[7,10] <= 1.45772e-05
 Conv_1_Relu_ss[7,11] <= 0.00236660145696522
 Conv_1_Relu_ss[7,12] <= 3.08689e-07
 Conv_1_Relu_ss[7,13] <= 4.431268808214668
 Conv_1_Relu_ss[7,14] <= 0.178791987482969
 Conv_1_Relu_ss[7,15] <= 1.22225e-07
 Conv_2_aggregated_features[0,0] = 0
 Conv_2_aggregated_features[0,1] <= 14.59003230898199
 Conv_2_aggregated_features[0,2] = 0
 Conv_2_aggregated_features[0,3] = 0
 Conv_2_aggregated_features[0,4] = 0
 Conv_2_aggregated_features[0,5] = 0
 Conv_2_aggregated_features[0,6] = 0
 Conv_2_aggregated_features[0,7] = 0
 Conv_2_aggregated_features[0,8] <= 2.75325e-13
 Conv_2_aggregated_features[0,9] <= 8.234055808249286
 Conv_2_aggregated_features[0,10] = 0
 Conv_2_aggregated_features[0,11] <= 3.07161e-05
 Conv_2_aggregated_features[0,12] = 0
 Conv_2_aggregated_features[0,13] <= 11.92005955570159
 Conv_2_aggregated_features[0,14] <= 0.00295504399667217
 Conv_2_aggregated_features[0,15] = 0
 Conv_2_aggregated_features[1,0] = 0
 Conv_2_aggregated_features[1,1] <= 14.59003230898199
 Conv_2_aggregated_features[1,2] = 0
 Conv_2_aggregated_features[1,3] = 0
 Conv_2_aggregated_features[1,4] = 0
 Conv_2_aggregated_features[1,5] = 0
 Conv_2_aggregated_features[1,6] = 0
 Conv_2_aggregated_features[1,7] = 0
 Conv_2_aggregated_features[1,8] <= 2.75325e-13
 Conv_2_aggregated_features[1,9] <= 8.234055808249286
 Conv_2_aggregated_features[1,10] = 0
 Conv_2_aggregated_features[1,11] <= 3.07161e-05
 Conv_2_aggregated_features[1,12] = 0
 Conv_2_aggregated_features[1,13] <= 11.92005955570159
 Conv_2_aggregated_features[1,14] <= 0.00295504399667217
 Conv_2_aggregated_features[1,15] = 0
 Conv_2_aggregated_features[2,0] = 0
 Conv_2_aggregated_features[2,1] <= 14.59003230898199
 Conv_2_aggregated_features[2,2] = 0
 Conv_2_aggregated_features[2,3] = 0
 Conv_2_aggregated_features[2,4] = 0
 Conv_2_aggregated_features[2,5] = 0
 Conv_2_aggregated_features[2,6] = 0
 Conv_2_aggregated_features[2,7] = 0
 Conv_2_aggregated_features[2,8] <= 2.75325e-13
 Conv_2_aggregated_features[2,9] <= 8.234055808249286
 Conv_2_aggregated_features[2,10] = 0
 Conv_2_aggregated_features[2,11] <= 3.07161e-05
 Conv_2_aggregated_features[2,12] = 0
 Conv_2_aggregated_features[2,13] <= 11.92005955570159
 Conv_2_aggregated_features[2,14] <= 0.00295504399667217
 Conv_2_aggregated_features[2,15] = 0
 Conv_2_aggregated_features[3,0] = 0
 Conv_2_aggregated_features[3,1] <= 14.59003230898199
 Conv_2_aggregated_features[3,2] = 0
 Conv_2_aggregated_features[3,3] = 0
 Conv_2_aggregated_features[3,4] = 0
 Conv_2_aggregated_features[3,5] = 0
 Conv_2_aggregated_features[3,6] = 0
 Conv_2_aggregated_features[3,7] = 0
 Conv_2_aggregated_features[3,8] <= 2.75325e-13
 Conv_2_aggregated_features[3,9] <= 8.234055808249286
 Conv_2_aggregated_features[3,10] = 0
 Conv_2_aggregated_features[3,11] <= 3.07161e-05
 Conv_2_aggregated_features[3,12] = 0
 Conv_2_aggregated_features[3,13] <= 11.92005955570159
 Conv_2_aggregated_features[3,14] <= 0.00295504399667217
 Conv_2_aggregated_features[3,15] = 0
 Conv_2_aggregated_features[4,0] = 0
 Conv_2_aggregated_features[4,1] <= 14.59003230898199
 Conv_2_aggregated_features[4,2] = 0
 Conv_2_aggregated_features[4,3] = 0
 Conv_2_aggregated_features[4,4] = 0
 Conv_2_aggregated_features[4,5] = 0
 Conv_2_aggregated_features[4,6] = 0
 Conv_2_aggregated_features[4,7] = 0
 Conv_2_aggregated_features[4,8] <= 2.75325e-13
 Conv_2_aggregated_features[4,9] <= 8.234055808249286
 Conv_2_aggregated_features[4,10] = 0
 Conv_2_aggregated_features[4,11] <= 3.07161e-05
 Conv_2_aggregated_features[4,12] = 0
 Conv_2_aggregated_features[4,13] <= 11.92005955570159
 Conv_2_aggregated_features[4,14] <= 0.00295504399667217
 Conv_2_aggregated_features[4,15] = 0
 Conv_2_aggregated_features[5,0] = 0
 Conv_2_aggregated_features[5,1] <= 14.59003230898199
 Conv_2_aggregated_features[5,2] = 0
 Conv_2_aggregated_features[5,3] = 0
 Conv_2_aggregated_features[5,4] = 0
 Conv_2_aggregated_features[5,5] = 0
 Conv_2_aggregated_features[5,6] = 0
 Conv_2_aggregated_features[5,7] = 0
 Conv_2_aggregated_features[5,8] <= 2.75325e-13
 Conv_2_aggregated_features[5,9] <= 8.234055808249286
 Conv_2_aggregated_features[5,10] = 0
 Conv_2_aggregated_features[5,11] <= 3.07161e-05
 Conv_2_aggregated_features[5,12] = 0
 Conv_2_aggregated_features[5,13] <= 11.92005955570159
 Conv_2_aggregated_features[5,14] <= 0.00295504399667217
 Conv_2_aggregated_features[5,15] = 0
 Conv_2_aggregated_features[6,0] = 0
 Conv_2_aggregated_features[6,1] <= 14.59003230898199
 Conv_2_aggregated_features[6,2] = 0
 Conv_2_aggregated_features[6,3] = 0
 Conv_2_aggregated_features[6,4] = 0
 Conv_2_aggregated_features[6,5] = 0
 Conv_2_aggregated_features[6,6] = 0
 Conv_2_aggregated_features[6,7] = 0
 Conv_2_aggregated_features[6,8] <= 2.75325e-13
 Conv_2_aggregated_features[6,9] <= 8.234055808249286
 Conv_2_aggregated_features[6,10] = 0
 Conv_2_aggregated_features[6,11] <= 3.07161e-05
 Conv_2_aggregated_features[6,12] = 0
 Conv_2_aggregated_features[6,13] <= 11.92005955570159
 Conv_2_aggregated_features[6,14] <= 0.00295504399667217
 Conv_2_aggregated_features[6,15] = 0
 Conv_2_aggregated_features[7,0] = 0
 Conv_2_aggregated_features[7,1] <= 14.59003230898199
 Conv_2_aggregated_features[7,2] = 0
 Conv_2_aggregated_features[7,3] = 0
 Conv_2_aggregated_features[7,4] = 0
 Conv_2_aggregated_features[7,5] = 0
 Conv_2_aggregated_features[7,6] = 0
 Conv_2_aggregated_features[7,7] = 0
 Conv_2_aggregated_features[7,8] <= 2.75325e-13
 Conv_2_aggregated_features[7,9] <= 8.234055808249286
 Conv_2_aggregated_features[7,10] = 0
 Conv_2_aggregated_features[7,11] <= 3.07161e-05
 Conv_2_aggregated_features[7,12] = 0
 Conv_2_aggregated_features[7,13] <= 11.92005955570159
 Conv_2_aggregated_features[7,14] <= 0.00295504399667217
 Conv_2_aggregated_features[7,15] = 0
 -2.60542e-12 <= Conv_2_t[0,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[0,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[0,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[0,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[0,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[0,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[0,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[0,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[0,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[0,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[0,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[0,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[0,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[0,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[0,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[0,15] <= 1.8051988252351864e-04
 -2.60542e-12 <= Conv_2_t[1,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[1,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[1,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[1,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[1,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[1,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[1,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[1,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[1,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[1,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[1,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[1,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[1,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[1,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[1,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[1,15] <= 1.8051988252351864e-04
 -2.60542e-12 <= Conv_2_t[2,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[2,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[2,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[2,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[2,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[2,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[2,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[2,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[2,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[2,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[2,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[2,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[2,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[2,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[2,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[2,15] <= 1.8051988252351864e-04
 -2.60542e-12 <= Conv_2_t[3,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[3,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[3,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[3,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[3,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[3,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[3,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[3,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[3,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[3,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[3,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[3,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[3,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[3,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[3,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[3,15] <= 1.8051988252351864e-04
 -2.60542e-12 <= Conv_2_t[4,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[4,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[4,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[4,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[4,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[4,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[4,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[4,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[4,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[4,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[4,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[4,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[4,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[4,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[4,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[4,15] <= 1.8051988252351864e-04
 -2.60542e-12 <= Conv_2_t[5,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[5,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[5,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[5,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[5,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[5,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[5,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[5,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[5,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[5,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[5,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[5,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[5,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[5,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[5,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[5,15] <= 1.8051988252351864e-04
 -2.60542e-12 <= Conv_2_t[6,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[6,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[6,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[6,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[6,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[6,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[6,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[6,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[6,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[6,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[6,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[6,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[6,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[6,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[6,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[6,15] <= 1.8051988252351864e-04
 -2.60542e-12 <= Conv_2_t[7,0] <= 5.88025e-13
 -0.00259094797606434 <= Conv_2_t[7,1] <= 4.5736554106570173e-04
 -3.60829e-10 <= Conv_2_t[7,2] <= -1.21231e-11
 4.02827e-07 <= Conv_2_t[7,3] <= 4.99422e-05
 -0.0176422467492259 <= Conv_2_t[7,4] <= 1.48705e-05
 -2.51833e-08 <= Conv_2_t[7,5] <= 6.86581e-09
 -38.08457690166041 <= Conv_2_t[7,6] <= 13.42537958453236
 -6.983479580766037 <= Conv_2_t[7,7] <= 11.98662083090122
 -1.9980672280076582e-04 <= Conv_2_t[7,8] <= 7.6174e-05
 2.10688e-05 <= Conv_2_t[7,9] <= 0.00112970281782641
 -4.8062e-07 <= Conv_2_t[7,10] <= -6.36497e-09
 3.19541e-09 <= Conv_2_t[7,11] <= 5.14323e-07
 -6.67338989035037 <= Conv_2_t[7,12] <= 19.55128642019131
 -0.7895277949390602 <= Conv_2_t[7,13] <= 0.8449266835182555
 0.3463285262768594 <= Conv_2_t[7,14] <= 17.02833982748556
 -0.000225001 <= Conv_2_t[7,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[0,0] <= 5.88025e-13
 Conv_2_Relu_ts[0,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[0,2] = 0
 Conv_2_Relu_ts[0,3] <= 4.99422e-05
 Conv_2_Relu_ts[0,4] <= 1.48705e-05
 Conv_2_Relu_ts[0,5] <= 6.86581e-09
 Conv_2_Relu_ts[0,6] <= 13.42537958453236
 Conv_2_Relu_ts[0,7] <= 11.98662083090122
 Conv_2_Relu_ts[0,8] <= 7.6174e-05
 Conv_2_Relu_ts[0,9] <= 0.00112970281782641
 Conv_2_Relu_ts[0,10] = 0
 Conv_2_Relu_ts[0,11] <= 5.14323e-07
 Conv_2_Relu_ts[0,12] <= 19.55128642019131
 Conv_2_Relu_ts[0,13] <= 0.8449266835182555
 Conv_2_Relu_ts[0,14] <= 17.02833982748556
 Conv_2_Relu_ts[0,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[1,0] <= 5.88025e-13
 Conv_2_Relu_ts[1,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[1,2] = 0
 Conv_2_Relu_ts[1,3] <= 4.99422e-05
 Conv_2_Relu_ts[1,4] <= 1.48705e-05
 Conv_2_Relu_ts[1,5] <= 6.86581e-09
 Conv_2_Relu_ts[1,6] <= 13.42537958453236
 Conv_2_Relu_ts[1,7] <= 11.98662083090122
 Conv_2_Relu_ts[1,8] <= 7.6174e-05
 Conv_2_Relu_ts[1,9] <= 0.00112970281782641
 Conv_2_Relu_ts[1,10] = 0
 Conv_2_Relu_ts[1,11] <= 5.14323e-07
 Conv_2_Relu_ts[1,12] <= 19.55128642019131
 Conv_2_Relu_ts[1,13] <= 0.8449266835182555
 Conv_2_Relu_ts[1,14] <= 17.02833982748556
 Conv_2_Relu_ts[1,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[2,0] <= 5.88025e-13
 Conv_2_Relu_ts[2,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[2,2] = 0
 Conv_2_Relu_ts[2,3] <= 4.99422e-05
 Conv_2_Relu_ts[2,4] <= 1.48705e-05
 Conv_2_Relu_ts[2,5] <= 6.86581e-09
 Conv_2_Relu_ts[2,6] <= 13.42537958453236
 Conv_2_Relu_ts[2,7] <= 11.98662083090122
 Conv_2_Relu_ts[2,8] <= 7.6174e-05
 Conv_2_Relu_ts[2,9] <= 0.00112970281782641
 Conv_2_Relu_ts[2,10] = 0
 Conv_2_Relu_ts[2,11] <= 5.14323e-07
 Conv_2_Relu_ts[2,12] <= 19.55128642019131
 Conv_2_Relu_ts[2,13] <= 0.8449266835182555
 Conv_2_Relu_ts[2,14] <= 17.02833982748556
 Conv_2_Relu_ts[2,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[3,0] <= 5.88025e-13
 Conv_2_Relu_ts[3,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[3,2] = 0
 Conv_2_Relu_ts[3,3] <= 4.99422e-05
 Conv_2_Relu_ts[3,4] <= 1.48705e-05
 Conv_2_Relu_ts[3,5] <= 6.86581e-09
 Conv_2_Relu_ts[3,6] <= 13.42537958453236
 Conv_2_Relu_ts[3,7] <= 11.98662083090122
 Conv_2_Relu_ts[3,8] <= 7.6174e-05
 Conv_2_Relu_ts[3,9] <= 0.00112970281782641
 Conv_2_Relu_ts[3,10] = 0
 Conv_2_Relu_ts[3,11] <= 5.14323e-07
 Conv_2_Relu_ts[3,12] <= 19.55128642019131
 Conv_2_Relu_ts[3,13] <= 0.8449266835182555
 Conv_2_Relu_ts[3,14] <= 17.02833982748556
 Conv_2_Relu_ts[3,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[4,0] <= 5.88025e-13
 Conv_2_Relu_ts[4,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[4,2] = 0
 Conv_2_Relu_ts[4,3] <= 4.99422e-05
 Conv_2_Relu_ts[4,4] <= 1.48705e-05
 Conv_2_Relu_ts[4,5] <= 6.86581e-09
 Conv_2_Relu_ts[4,6] <= 13.42537958453236
 Conv_2_Relu_ts[4,7] <= 11.98662083090122
 Conv_2_Relu_ts[4,8] <= 7.6174e-05
 Conv_2_Relu_ts[4,9] <= 0.00112970281782641
 Conv_2_Relu_ts[4,10] = 0
 Conv_2_Relu_ts[4,11] <= 5.14323e-07
 Conv_2_Relu_ts[4,12] <= 19.55128642019131
 Conv_2_Relu_ts[4,13] <= 0.8449266835182555
 Conv_2_Relu_ts[4,14] <= 17.02833982748556
 Conv_2_Relu_ts[4,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[5,0] <= 5.88025e-13
 Conv_2_Relu_ts[5,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[5,2] = 0
 Conv_2_Relu_ts[5,3] <= 4.99422e-05
 Conv_2_Relu_ts[5,4] <= 1.48705e-05
 Conv_2_Relu_ts[5,5] <= 6.86581e-09
 Conv_2_Relu_ts[5,6] <= 13.42537958453236
 Conv_2_Relu_ts[5,7] <= 11.98662083090122
 Conv_2_Relu_ts[5,8] <= 7.6174e-05
 Conv_2_Relu_ts[5,9] <= 0.00112970281782641
 Conv_2_Relu_ts[5,10] = 0
 Conv_2_Relu_ts[5,11] <= 5.14323e-07
 Conv_2_Relu_ts[5,12] <= 19.55128642019131
 Conv_2_Relu_ts[5,13] <= 0.8449266835182555
 Conv_2_Relu_ts[5,14] <= 17.02833982748556
 Conv_2_Relu_ts[5,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[6,0] <= 5.88025e-13
 Conv_2_Relu_ts[6,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[6,2] = 0
 Conv_2_Relu_ts[6,3] <= 4.99422e-05
 Conv_2_Relu_ts[6,4] <= 1.48705e-05
 Conv_2_Relu_ts[6,5] <= 6.86581e-09
 Conv_2_Relu_ts[6,6] <= 13.42537958453236
 Conv_2_Relu_ts[6,7] <= 11.98662083090122
 Conv_2_Relu_ts[6,8] <= 7.6174e-05
 Conv_2_Relu_ts[6,9] <= 0.00112970281782641
 Conv_2_Relu_ts[6,10] = 0
 Conv_2_Relu_ts[6,11] <= 5.14323e-07
 Conv_2_Relu_ts[6,12] <= 19.55128642019131
 Conv_2_Relu_ts[6,13] <= 0.8449266835182555
 Conv_2_Relu_ts[6,14] <= 17.02833982748556
 Conv_2_Relu_ts[6,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ts[7,0] <= 5.88025e-13
 Conv_2_Relu_ts[7,1] <= 4.5736554106570173e-04
 Conv_2_Relu_ts[7,2] = 0
 Conv_2_Relu_ts[7,3] <= 4.99422e-05
 Conv_2_Relu_ts[7,4] <= 1.48705e-05
 Conv_2_Relu_ts[7,5] <= 6.86581e-09
 Conv_2_Relu_ts[7,6] <= 13.42537958453236
 Conv_2_Relu_ts[7,7] <= 11.98662083090122
 Conv_2_Relu_ts[7,8] <= 7.6174e-05
 Conv_2_Relu_ts[7,9] <= 0.00112970281782641
 Conv_2_Relu_ts[7,10] = 0
 Conv_2_Relu_ts[7,11] <= 5.14323e-07
 Conv_2_Relu_ts[7,12] <= 19.55128642019131
 Conv_2_Relu_ts[7,13] <= 0.8449266835182555
 Conv_2_Relu_ts[7,14] <= 17.02833982748556
 Conv_2_Relu_ts[7,15] <= 1.8051988252351864e-04
 Conv_2_Relu_ss[0,0] <= 2.60542e-12
 Conv_2_Relu_ss[0,1] <= 0.00259094797606434
 Conv_2_Relu_ss[0,2] <= 3.60829e-10
 Conv_2_Relu_ss[0,3] = 0
 Conv_2_Relu_ss[0,4] <= 0.0176422467492259
 Conv_2_Relu_ss[0,5] <= 2.51833e-08
 Conv_2_Relu_ss[0,6] <= 38.08457690166041
 Conv_2_Relu_ss[0,7] <= 6.983479580766037
 Conv_2_Relu_ss[0,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[0,9] = 0
 Conv_2_Relu_ss[0,10] <= 4.8062e-07
 Conv_2_Relu_ss[0,11] = 0
 Conv_2_Relu_ss[0,12] <= 6.67338989035037
 Conv_2_Relu_ss[0,13] <= 0.7895277949390602
 Conv_2_Relu_ss[0,14] = 0
 Conv_2_Relu_ss[0,15] <= 0.000225001
 Conv_2_Relu_ss[1,0] <= 2.60542e-12
 Conv_2_Relu_ss[1,1] <= 0.00259094797606434
 Conv_2_Relu_ss[1,2] <= 3.60829e-10
 Conv_2_Relu_ss[1,3] = 0
 Conv_2_Relu_ss[1,4] <= 0.0176422467492259
 Conv_2_Relu_ss[1,5] <= 2.51833e-08
 Conv_2_Relu_ss[1,6] <= 38.08457690166041
 Conv_2_Relu_ss[1,7] <= 6.983479580766037
 Conv_2_Relu_ss[1,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[1,9] = 0
 Conv_2_Relu_ss[1,10] <= 4.8062e-07
 Conv_2_Relu_ss[1,11] = 0
 Conv_2_Relu_ss[1,12] <= 6.67338989035037
 Conv_2_Relu_ss[1,13] <= 0.7895277949390602
 Conv_2_Relu_ss[1,14] = 0
 Conv_2_Relu_ss[1,15] <= 0.000225001
 Conv_2_Relu_ss[2,0] <= 2.60542e-12
 Conv_2_Relu_ss[2,1] <= 0.00259094797606434
 Conv_2_Relu_ss[2,2] <= 3.60829e-10
 Conv_2_Relu_ss[2,3] = 0
 Conv_2_Relu_ss[2,4] <= 0.0176422467492259
 Conv_2_Relu_ss[2,5] <= 2.51833e-08
 Conv_2_Relu_ss[2,6] <= 38.08457690166041
 Conv_2_Relu_ss[2,7] <= 6.983479580766037
 Conv_2_Relu_ss[2,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[2,9] = 0
 Conv_2_Relu_ss[2,10] <= 4.8062e-07
 Conv_2_Relu_ss[2,11] = 0
 Conv_2_Relu_ss[2,12] <= 6.67338989035037
 Conv_2_Relu_ss[2,13] <= 0.7895277949390602
 Conv_2_Relu_ss[2,14] = 0
 Conv_2_Relu_ss[2,15] <= 0.000225001
 Conv_2_Relu_ss[3,0] <= 2.60542e-12
 Conv_2_Relu_ss[3,1] <= 0.00259094797606434
 Conv_2_Relu_ss[3,2] <= 3.60829e-10
 Conv_2_Relu_ss[3,3] = 0
 Conv_2_Relu_ss[3,4] <= 0.0176422467492259
 Conv_2_Relu_ss[3,5] <= 2.51833e-08
 Conv_2_Relu_ss[3,6] <= 38.08457690166041
 Conv_2_Relu_ss[3,7] <= 6.983479580766037
 Conv_2_Relu_ss[3,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[3,9] = 0
 Conv_2_Relu_ss[3,10] <= 4.8062e-07
 Conv_2_Relu_ss[3,11] = 0
 Conv_2_Relu_ss[3,12] <= 6.67338989035037
 Conv_2_Relu_ss[3,13] <= 0.7895277949390602
 Conv_2_Relu_ss[3,14] = 0
 Conv_2_Relu_ss[3,15] <= 0.000225001
 Conv_2_Relu_ss[4,0] <= 2.60542e-12
 Conv_2_Relu_ss[4,1] <= 0.00259094797606434
 Conv_2_Relu_ss[4,2] <= 3.60829e-10
 Conv_2_Relu_ss[4,3] = 0
 Conv_2_Relu_ss[4,4] <= 0.0176422467492259
 Conv_2_Relu_ss[4,5] <= 2.51833e-08
 Conv_2_Relu_ss[4,6] <= 38.08457690166041
 Conv_2_Relu_ss[4,7] <= 6.983479580766037
 Conv_2_Relu_ss[4,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[4,9] = 0
 Conv_2_Relu_ss[4,10] <= 4.8062e-07
 Conv_2_Relu_ss[4,11] = 0
 Conv_2_Relu_ss[4,12] <= 6.67338989035037
 Conv_2_Relu_ss[4,13] <= 0.7895277949390602
 Conv_2_Relu_ss[4,14] = 0
 Conv_2_Relu_ss[4,15] <= 0.000225001
 Conv_2_Relu_ss[5,0] <= 2.60542e-12
 Conv_2_Relu_ss[5,1] <= 0.00259094797606434
 Conv_2_Relu_ss[5,2] <= 3.60829e-10
 Conv_2_Relu_ss[5,3] = 0
 Conv_2_Relu_ss[5,4] <= 0.0176422467492259
 Conv_2_Relu_ss[5,5] <= 2.51833e-08
 Conv_2_Relu_ss[5,6] <= 38.08457690166041
 Conv_2_Relu_ss[5,7] <= 6.983479580766037
 Conv_2_Relu_ss[5,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[5,9] = 0
 Conv_2_Relu_ss[5,10] <= 4.8062e-07
 Conv_2_Relu_ss[5,11] = 0
 Conv_2_Relu_ss[5,12] <= 6.67338989035037
 Conv_2_Relu_ss[5,13] <= 0.7895277949390602
 Conv_2_Relu_ss[5,14] = 0
 Conv_2_Relu_ss[5,15] <= 0.000225001
 Conv_2_Relu_ss[6,0] <= 2.60542e-12
 Conv_2_Relu_ss[6,1] <= 0.00259094797606434
 Conv_2_Relu_ss[6,2] <= 3.60829e-10
 Conv_2_Relu_ss[6,3] = 0
 Conv_2_Relu_ss[6,4] <= 0.0176422467492259
 Conv_2_Relu_ss[6,5] <= 2.51833e-08
 Conv_2_Relu_ss[6,6] <= 38.08457690166041
 Conv_2_Relu_ss[6,7] <= 6.983479580766037
 Conv_2_Relu_ss[6,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[6,9] = 0
 Conv_2_Relu_ss[6,10] <= 4.8062e-07
 Conv_2_Relu_ss[6,11] = 0
 Conv_2_Relu_ss[6,12] <= 6.67338989035037
 Conv_2_Relu_ss[6,13] <= 0.7895277949390602
 Conv_2_Relu_ss[6,14] = 0
 Conv_2_Relu_ss[6,15] <= 0.000225001
 Conv_2_Relu_ss[7,0] <= 2.60542e-12
 Conv_2_Relu_ss[7,1] <= 0.00259094797606434
 Conv_2_Relu_ss[7,2] <= 3.60829e-10
 Conv_2_Relu_ss[7,3] = 0
 Conv_2_Relu_ss[7,4] <= 0.0176422467492259
 Conv_2_Relu_ss[7,5] <= 2.51833e-08
 Conv_2_Relu_ss[7,6] <= 38.08457690166041
 Conv_2_Relu_ss[7,7] <= 6.983479580766037
 Conv_2_Relu_ss[7,8] <= 1.9980672280076582e-04
 Conv_2_Relu_ss[7,9] = 0
 Conv_2_Relu_ss[7,10] <= 4.8062e-07
 Conv_2_Relu_ss[7,11] = 0
 Conv_2_Relu_ss[7,12] <= 6.67338989035037
 Conv_2_Relu_ss[7,13] <= 0.7895277949390602
 Conv_2_Relu_ss[7,14] = 0
 Conv_2_Relu_ss[7,15] <= 0.000225001
 Aggregation[0] <= 5.88025e-13
 Aggregation[1] <= 4.5736554106570173e-04
 Aggregation[2] = 0
 Aggregation[3] <= 4.99422e-05
 Aggregation[4] <= 1.48705e-05
 Aggregation[5] <= 6.86581e-09
 Aggregation[6] <= 13.42537958453236
 Aggregation[7] <= 11.98662083090122
 Aggregation[8] <= 7.6174e-05
 Aggregation[9] <= 0.00112970281782641
 Aggregation[10] = 0
 Aggregation[11] <= 5.14323e-07
 Aggregation[12] <= 19.55128642019131
 Aggregation[13] <= 0.8449266835182555
 Aggregation[14] <= 17.02833982748556
 Aggregation[15] <= 1.8051988252351864e-04
 -0.1120404087063036 <= Lin_0_t[0] <= 0.0646826258068006
 -15.56777676550227 <= Lin_0_t[1] <= 23.48739610133581
 -6.981702896342312 <= Lin_0_t[2] <= 15.10090508869101
 -11.59974143443083 <= Lin_0_t[3] <= 18.36376828199457
 -5.941669058398669 <= Lin_0_t[4] <= 28.14687881992175
 -2.896736603265128 <= Lin_0_t[5] <= 3.366320737461283
 -4.6017981019739996e-04 <= Lin_0_t[6] <= 0.002196731845088
 -3.673082476382535 <= Lin_0_t[7] <= 2.918435776786758
 Lin_0_Relu_ts[0,0] <= 0.0646826258068006
 Lin_0_Relu_ts[0,1] <= 23.48739610133581
 Lin_0_Relu_ts[0,2] <= 15.10090508869101
 Lin_0_Relu_ts[0,3] <= 18.36376828199457
 Lin_0_Relu_ts[0,4] <= 28.14687881992175
 Lin_0_Relu_ts[0,5] <= 3.366320737461283
 Lin_0_Relu_ts[0,6] <= 0.002196731845088
 Lin_0_Relu_ts[0,7] <= 2.918435776786758
 Lin_0_Relu_ss[0,0] <= 0.1120404087063036
 Lin_0_Relu_ss[0,1] <= 15.56777676550227
 Lin_0_Relu_ss[0,2] <= 6.981702896342312
 Lin_0_Relu_ss[0,3] <= 11.59974143443083
 Lin_0_Relu_ss[0,4] <= 5.941669058398669
 Lin_0_Relu_ss[0,5] <= 2.896736603265128
 Lin_0_Relu_ss[0,6] <= 4.6017981019739996e-04
 Lin_0_Relu_ss[0,7] <= 3.673082476382535
 -13.82460327817286 <= Lin_1_t[0] <= 32.57844495195941
 -2.3286370404764788e-04 <= Lin_1_t[1] <= 6.46128e-05
 -21.83153581211795 <= Lin_1_t[2] <= 30.92121960101973
 -8.962141052203886 <= Lin_1_t[3] <= 7.233531117990599
 -13.10294566735477 <= Lin_1_t[4] <= 23.21907216629517
 -14.56961657055903 <= Lin_1_t[5] <= 12.32466612086686
 -2.59718e-07 <= Lin_1_t[6] <= 8.26732e-06
 -29.85479667112976 <= Lin_1_t[7] <= 38.97431296201979
 Lin1Relu_ts[0,0] <= 32.57844495195941
 Lin1Relu_ts[0,1] <= 6.46128e-05
 Lin1Relu_ts[0,2] <= 30.92121960101973
 Lin1Relu_ts[0,3] <= 7.233531117990599
 Lin1Relu_ts[0,4] <= 23.21907216629517
 Lin1Relu_ts[0,5] <= 12.32466612086686
 Lin1Relu_ts[0,6] <= 8.26732e-06
 Lin1Relu_ts[0,7] <= 38.97431296201979
 Lin1Relu_ss[0,0] <= 13.82460327817286
 Lin1Relu_ss[0,1] <= 2.3286370404764788e-04
 Lin1Relu_ss[0,2] <= 21.83153581211795
 Lin1Relu_ss[0,3] <= 8.962141052203886
 Lin1Relu_ss[0,4] <= 13.10294566735477
 Lin1Relu_ss[0,5] <= 14.56961657055903
 Lin1Relu_ss[0,6] <= 2.59718e-07
 Lin1Relu_ss[0,7] <= 29.85479667112976
 -67.50530288903515 <= Output_t[0] <= 34.5877260773664
 -28.45265368522516 <= Output_t[1] <= 48.42399425795631
 -25.83989739821254 <= Output_t[2] <= 38.0603177623837
 -45.13032090740785 <= Output_t[3] <= 29.24916971274846
 -13.46841791680866 <= Output_t[4] <= 22.37055431417357
 embedding_similarity <= 1
 embedding_magnitude <= 31.57294659236448
Binaries
 A[0,0] A[0,1] A[0,2] A[0,3] A[0,4] A[0,5] A[0,6] A[0,7] A[1,0] A[1,1]
 A[1,2] A[1,3] A[1,4] A[1,5] A[1,6] A[1,7] A[2,0] A[2,1] A[2,2] A[2,3]
 A[2,4] A[2,5] A[2,6] A[2,7] A[3,0] A[3,1] A[3,2] A[3,3] A[3,4] A[3,5]
 A[3,6] A[3,7] A[4,0] A[4,1] A[4,2] A[4,3] A[4,4] A[4,5] A[4,6] A[4,7]
 A[5,0] A[5,1] A[5,2] A[5,3] A[5,4] A[5,5] A[5,6] A[5,7] A[6,0] A[6,1]
 A[6,2] A[6,3] A[6,4] A[6,5] A[6,6] A[6,7] A[7,0] A[7,1] A[7,2] A[7,3]
 A[7,4] A[7,5] A[7,6] A[7,7] C464 C465 C466 C467 C468 C469 C470 C471 C472
 C473 C474 C475 C476 C477 C478 C479 C480 C481 C482 C483 C484 C485 C486 C487
 C488 C489 C490 C491 C492 C493 C494 C495 C496 C497 C498 C499 C500 C501 C502
 C503 C504 C505 C506 C507 C508 C509 C510 C511 C512 C513 C514 C515 C516 C517
 C518 C519 C520 C521 C522 C523 C524 C525 C526 C527 C528 C529 C530 C531 C532
 C533 C534 C535 C536 C537 C538 C539 C540 C541 C542 C543 C544 C545 C546 C547
 C548 C549 C550 C551 C552 C553 C554 C555 C556 C557 C558 C559 C560 C561 C562
 C563 C564 C565 C566 C567 C568 C569 C570 C571 C572 C573 C574 C575 C576 C577
 C578 C579 C580 C581 C582 C583 C584 C585 C586 C587 C588 C589 C590 C591
 C1104 C1105 C1106 C1107 C1108 C1109 C1110 C1111 C1112 C1113 C1114 C1115
 C1116 C1117 C1118 C1119 C1120 C1121 C1122 C1123 C1124 C1125 C1126 C1127
 C1128 C1129 C1130 C1131 C1132 C1133 C1134 C1135 C1136 C1137 C1138 C1139
 C1140 C1141 C1142 C1143 C1144 C1145 C1146 C1147 C1148 C1149 C1150 C1151
 C1152 C1153 C1154 C1155 C1156 C1157 C1158 C1159 C1160 C1161 C1162 C1163
 C1164 C1165 C1166 C1167 C1168 C1169 C1170 C1171 C1172 C1173 C1174 C1175
 C1176 C1177 C1178 C1179 C1180 C1181 C1182 C1183 C1184 C1185 C1186 C1187
 C1188 C1189 C1190 C1191 C1192 C1193 C1194 C1195 C1196 C1197 C1198 C1199
 C1200 C1201 C1202 C1203 C1204 C1205 C1206 C1207 C1208 C1209 C1210 C1211
 C1212 C1213 C1214 C1215 C1216 C1217 C1218 C1219 C1220 C1221 C1222 C1223
 C1224 C1225 C1226 C1227 C1228 C1229 C1230 C1231 C1744 C1745 C1746 C1747
 C1748 C1749 C1750 C1751 C1752 C1753 C1754 C1755 C1756 C1757 C1758 C1759
 C1760 C1761 C1762 C1763 C1764 C1765 C1766 C1767 C1768 C1769 C1770 C1771
 C1772 C1773 C1774 C1775 C1776 C1777 C1778 C1779 C1780 C1781 C1782 C1783
 C1784 C1785 C1786 C1787 C1788 C1789 C1790 C1791 C1792 C1793 C1794 C1795
 C1796 C1797 C1798 C1799 C1800 C1801 C1802 C1803 C1804 C1805 C1806 C1807
 C1808 C1809 C1810 C1811 C1812 C1813 C1814 C1815 C1816 C1817 C1818 C1819
 C1820 C1821 C1822 C1823 C1824 C1825 C1826 C1827 C1828 C1829 C1830 C1831
 C1832 C1833 C1834 C1835 C1836 C1837 C1838 C1839 C1840 C1841 C1842 C1843
 C1844 C1845 C1846 C1847 C1848 C1849 C1850 C1851 C1852 C1853 C1854 C1855
 C1856 C1857 C1858 C1859 C1860 C1861 C1862 C1863 C1864 C1865 C1866 C1867
 C1868 C1869 C1870 C1871 C1912 C1913 C1914 C1915 C1916 C1917 C1918 C1919
 C1944 C1945 C1946 C1947 C1948 C1949 C1950 C1951
Generals
 X[0,0] X[1,0] X[2,0] X[3,0] X[4,0] X[5,0] X[6,0] X[7,0]
General Constraints
 embedding_magnitude_constraint: embedding_magnitude = NORM ( 2 ) (
   Aggregation[0] , Aggregation[1] , Aggregation[2] , Aggregation[3] ,
   Aggregation[4] , Aggregation[5] , Aggregation[6] , Aggregation[7] ,
   Aggregation[8] , Aggregation[9] , Aggregation[10] , Aggregation[11] ,
   Aggregation[12] , Aggregation[13] , Aggregation[14] , Aggregation[15] )
End
