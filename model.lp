\ Model GNN Inverse
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  Output_t[0] + 10 embedding_similarity
Subject To
 node_0_connected: A[0,1] + A[0,2] + A[0,3] + A[0,4] + A[0,5] + A[0,6]
   + A[0,7] + A[0,8] + A[0,9] + A[1,0] + A[2,0] + A[3,0] + A[4,0] + A[5,0]
   + A[6,0] + A[7,0] + A[8,0] + A[9,0] >= 1
 node_1_connected: A[1,2] + A[1,3] + A[1,4] + A[1,5] + A[1,6] + A[1,7]
   + A[1,8] + A[1,9] + A[2,1] + A[3,1] + A[4,1] + A[5,1] + A[6,1] + A[7,1]
   + A[8,1] + A[9,1] >= 1
 node_2_connected: A[2,3] + A[2,4] + A[2,5] + A[2,6] + A[2,7] + A[2,8]
   + A[2,9] + A[3,2] + A[4,2] + A[5,2] + A[6,2] + A[7,2] + A[8,2] + A[9,2]
   >= 1
 node_3_connected: A[3,4] + A[3,5] + A[3,6] + A[3,7] + A[3,8] + A[3,9]
   + A[4,3] + A[5,3] + A[6,3] + A[7,3] + A[8,3] + A[9,3] >= 1
 node_4_connected: A[4,5] + A[4,6] + A[4,7] + A[4,8] + A[4,9] + A[5,4]
   + A[6,4] + A[7,4] + A[8,4] + A[9,4] >= 1
 node_5_connected: A[5,6] + A[5,7] + A[5,8] + A[5,9] + A[6,5] + A[7,5]
   + A[8,5] + A[9,5] >= 1
 node_6_connected: A[6,7] + A[6,8] + A[6,9] + A[7,6] + A[8,6] + A[9,6]
   >= 1
 node_7_connected: A[7,8] + A[7,9] + A[8,7] + A[9,7] >= 1
 node_8_connected: A[8,9] + A[9,8] >= 1
 undirected_0_0: = 0
 undirected_0_1: A[0,1] - A[1,0] = 0
 undirected_0_2: A[0,2] - A[2,0] = 0
 undirected_0_3: A[0,3] - A[3,0] = 0
 undirected_0_4: A[0,4] - A[4,0] = 0
 undirected_0_5: A[0,5] - A[5,0] = 0
 undirected_0_6: A[0,6] - A[6,0] = 0
 undirected_0_7: A[0,7] - A[7,0] = 0
 undirected_0_8: A[0,8] - A[8,0] = 0
 undirected_0_9: A[0,9] - A[9,0] = 0
 undirected_1_1: = 0
 undirected_1_2: A[1,2] - A[2,1] = 0
 undirected_1_3: A[1,3] - A[3,1] = 0
 undirected_1_4: A[1,4] - A[4,1] = 0
 undirected_1_5: A[1,5] - A[5,1] = 0
 undirected_1_6: A[1,6] - A[6,1] = 0
 undirected_1_7: A[1,7] - A[7,1] = 0
 undirected_1_8: A[1,8] - A[8,1] = 0
 undirected_1_9: A[1,9] - A[9,1] = 0
 undirected_2_2: = 0
 undirected_2_3: A[2,3] - A[3,2] = 0
 undirected_2_4: A[2,4] - A[4,2] = 0
 undirected_2_5: A[2,5] - A[5,2] = 0
 undirected_2_6: A[2,6] - A[6,2] = 0
 undirected_2_7: A[2,7] - A[7,2] = 0
 undirected_2_8: A[2,8] - A[8,2] = 0
 undirected_2_9: A[2,9] - A[9,2] = 0
 undirected_3_3: = 0
 undirected_3_4: A[3,4] - A[4,3] = 0
 undirected_3_5: A[3,5] - A[5,3] = 0
 undirected_3_6: A[3,6] - A[6,3] = 0
 undirected_3_7: A[3,7] - A[7,3] = 0
 undirected_3_8: A[3,8] - A[8,3] = 0
 undirected_3_9: A[3,9] - A[9,3] = 0
 undirected_4_4: = 0
 undirected_4_5: A[4,5] - A[5,4] = 0
 undirected_4_6: A[4,6] - A[6,4] = 0
 undirected_4_7: A[4,7] - A[7,4] = 0
 undirected_4_8: A[4,8] - A[8,4] = 0
 undirected_4_9: A[4,9] - A[9,4] = 0
 undirected_5_5: = 0
 undirected_5_6: A[5,6] - A[6,5] = 0
 undirected_5_7: A[5,7] - A[7,5] = 0
 undirected_5_8: A[5,8] - A[8,5] = 0
 undirected_5_9: A[5,9] - A[9,5] = 0
 undirected_6_6: = 0
 undirected_6_7: A[6,7] - A[7,6] = 0
 undirected_6_8: A[6,8] - A[8,6] = 0
 undirected_6_9: A[6,9] - A[9,6] = 0
 undirected_7_7: = 0
 undirected_7_8: A[7,8] - A[8,7] = 0
 undirected_7_9: A[7,9] - A[9,7] = 0
 undirected_8_8: = 0
 undirected_8_9: A[8,9] - A[9,8] = 0
 undirected_9_9: = 0
 non_isolatied[0]: A[0,0] + A[1,0] + A[2,0] + A[3,0] + A[4,0] + A[5,0]
   + A[6,0] + A[7,0] + A[8,0] + A[9,0] >= 1
 non_isolatied[1]: A[0,1] + A[1,1] + A[2,1] + A[3,1] + A[4,1] + A[5,1]
   + A[6,1] + A[7,1] + A[8,1] + A[9,1] >= 1
 non_isolatied[2]: A[0,2] + A[1,2] + A[2,2] + A[3,2] + A[4,2] + A[5,2]
   + A[6,2] + A[7,2] + A[8,2] + A[9,2] >= 1
 non_isolatied[3]: A[0,3] + A[1,3] + A[2,3] + A[3,3] + A[4,3] + A[5,3]
   + A[6,3] + A[7,3] + A[8,3] + A[9,3] >= 1
 non_isolatied[4]: A[0,4] + A[1,4] + A[2,4] + A[3,4] + A[4,4] + A[5,4]
   + A[6,4] + A[7,4] + A[8,4] + A[9,4] >= 1
 non_isolatied[5]: A[0,5] + A[1,5] + A[2,5] + A[3,5] + A[4,5] + A[5,5]
   + A[6,5] + A[7,5] + A[8,5] + A[9,5] >= 1
 non_isolatied[6]: A[0,6] + A[1,6] + A[2,6] + A[3,6] + A[4,6] + A[5,6]
   + A[6,6] + A[7,6] + A[8,6] + A[9,6] >= 1
 non_isolatied[7]: A[0,7] + A[1,7] + A[2,7] + A[3,7] + A[4,7] + A[5,7]
   + A[6,7] + A[7,7] + A[8,7] + A[9,7] >= 1
 non_isolatied[8]: A[0,8] + A[1,8] + A[2,8] + A[3,8] + A[4,8] + A[5,8]
   + A[6,8] + A[7,8] + A[8,8] + A[9,8] >= 1
 non_isolatied[9]: A[0,9] + A[1,9] + A[2,9] + A[3,9] + A[4,9] + A[5,9]
   + A[6,9] + A[7,9] + A[8,9] + A[9,9] >= 1
 features_are_node_degrees[0,0]: - A[0,0] - A[1,0] - A[2,0] - A[3,0]
   - A[4,0] - A[5,0] - A[6,0] - A[7,0] - A[8,0] - A[9,0] + X[0,0] = 0
 features_are_node_degrees[1,0]: - A[0,1] - A[1,1] - A[2,1] - A[3,1]
   - A[4,1] - A[5,1] - A[6,1] - A[7,1] - A[8,1] - A[9,1] + X[1,0] = 0
 features_are_node_degrees[2,0]: - A[0,2] - A[1,2] - A[2,2] - A[3,2]
   - A[4,2] - A[5,2] - A[6,2] - A[7,2] - A[8,2] - A[9,2] + X[2,0] = 0
 features_are_node_degrees[3,0]: - A[0,3] - A[1,3] - A[2,3] - A[3,3]
   - A[4,3] - A[5,3] - A[6,3] - A[7,3] - A[8,3] - A[9,3] + X[3,0] = 0
 features_are_node_degrees[4,0]: - A[0,4] - A[1,4] - A[2,4] - A[3,4]
   - A[4,4] - A[5,4] - A[6,4] - A[7,4] - A[8,4] - A[9,4] + X[4,0] = 0
 features_are_node_degrees[5,0]: - A[0,5] - A[1,5] - A[2,5] - A[3,5]
   - A[4,5] - A[5,5] - A[6,5] - A[7,5] - A[8,5] - A[9,5] + X[5,0] = 0
 features_are_node_degrees[6,0]: - A[0,6] - A[1,6] - A[2,6] - A[3,6]
   - A[4,6] - A[5,6] - A[6,6] - A[7,6] - A[8,6] - A[9,6] + X[6,0] = 0
 features_are_node_degrees[7,0]: - A[0,7] - A[1,7] - A[2,7] - A[3,7]
   - A[4,7] - A[5,7] - A[6,7] - A[7,7] - A[8,7] - A[9,7] + X[7,0] = 0
 features_are_node_degrees[8,0]: - A[0,8] - A[1,8] - A[2,8] - A[3,8]
   - A[4,8] - A[5,8] - A[6,8] - A[7,8] - A[8,8] - A[9,8] + X[8,0] = 0
 features_are_node_degrees[9,0]: - A[0,9] - A[1,9] - A[2,9] - A[3,9]
   - A[4,9] - A[5,9] - A[6,9] - A[7,9] - A[8,9] - A[9,9] + X[9,0] = 0
 Conv_0_output_constraint[0,0]: 0.2364515662193298 X[0,0]
   - 0.00366021576337516 Conv_0_aggregated_features[0,0] + Conv_0_t[0,0]
   = -0.8696585297584534
 Conv_0_output_constraint[0,1]: - 1.855180263519287 X[0,0]
   + 1.083907604217529 Conv_0_aggregated_features[0,0] + Conv_0_t[0,1]
   = -0.9598205089569092
 Conv_0_output_constraint[0,2]: 1.485349774360657 X[0,0]
   + 0.046770416200161 Conv_0_aggregated_features[0,0] + Conv_0_t[0,2]
   = -1.321228504180908
 Conv_0_output_constraint[0,3]: 0.7752511501312256 X[0,0]
   + 0.6645643711090088 Conv_0_aggregated_features[0,0] + Conv_0_t[0,3]
   = -0.4734632968902588
 Conv_0_output_constraint[0,4]: 1.752691864967346 X[0,0]
   + 0.0825358927249908 Conv_0_aggregated_features[0,0] + Conv_0_t[0,4]
   = 1.639361500740051
 Conv_0_output_constraint[0,5]: 1.845853567123413 X[0,0]
   - 0.1997903436422348 Conv_0_aggregated_features[0,0] + Conv_0_t[0,5]
   = 2.803231716156006
 Conv_0_output_constraint[0,6]: 0.5099456906318665 X[0,0]
   + 1.153847813606262 Conv_0_aggregated_features[0,0] + Conv_0_t[0,6]
   = 2.349466323852539
 Conv_0_output_constraint[0,7]: 1.624484419822693 X[0,0]
   + 0.069025494158268 Conv_0_aggregated_features[0,0] + Conv_0_t[0,7]
   = 0.9157499670982361
 Conv_0_output_constraint[0,8]: 0.8975760936737061 X[0,0]
   + 0.0988729 Conv_0_aggregated_features[0,0] + Conv_0_t[0,8]
   = -0.059127077460289
 Conv_0_output_constraint[0,9]: 1.157573580741882 X[0,0]
   + 0.00635915342718363 Conv_0_aggregated_features[0,0] + Conv_0_t[0,9]
   = -1.259021520614624
 Conv_0_output_constraint[0,10]: 1.51606810092926 X[0,0]
   + 0.1159021332859993 Conv_0_aggregated_features[0,0] + Conv_0_t[0,10]
   = -1.292696356773376
 Conv_0_output_constraint[0,11]: - 0.3291687667369843 X[0,0]
   + 1.914036870002747 Conv_0_aggregated_features[0,0] + Conv_0_t[0,11]
   = 1.365784168243408
 Conv_0_output_constraint[0,12]: - 0.3346672058105469 X[0,0]
   + 0.311100959777832 Conv_0_aggregated_features[0,0] + Conv_0_t[0,12]
   = -3.282561302185059
 Conv_0_output_constraint[0,13]: 0.9488325715065002 X[0,0]
   + 0.2108359783887863 Conv_0_aggregated_features[0,0] + Conv_0_t[0,13]
   = 3.006435632705688
 Conv_0_output_constraint[0,14]: 1.274890184402466 X[0,0]
   - 0.7632738351821899 Conv_0_aggregated_features[0,0] + Conv_0_t[0,14]
   = -2.962445259094238
 Conv_0_output_constraint[0,15]: 0.0870685502886772 X[0,0]
   + 0.1614511907100677 Conv_0_aggregated_features[0,0] + Conv_0_t[0,15]
   = -1.766824722290039
 Conv_0_output_constraint[1,0]: 0.2364515662193298 X[1,0]
   - 0.00366021576337516 Conv_0_aggregated_features[1,0] + Conv_0_t[1,0]
   = -0.8696585297584534
 Conv_0_output_constraint[1,1]: - 1.855180263519287 X[1,0]
   + 1.083907604217529 Conv_0_aggregated_features[1,0] + Conv_0_t[1,1]
   = -0.9598205089569092
 Conv_0_output_constraint[1,2]: 1.485349774360657 X[1,0]
   + 0.046770416200161 Conv_0_aggregated_features[1,0] + Conv_0_t[1,2]
   = -1.321228504180908
 Conv_0_output_constraint[1,3]: 0.7752511501312256 X[1,0]
   + 0.6645643711090088 Conv_0_aggregated_features[1,0] + Conv_0_t[1,3]
   = -0.4734632968902588
 Conv_0_output_constraint[1,4]: 1.752691864967346 X[1,0]
   + 0.0825358927249908 Conv_0_aggregated_features[1,0] + Conv_0_t[1,4]
   = 1.639361500740051
 Conv_0_output_constraint[1,5]: 1.845853567123413 X[1,0]
   - 0.1997903436422348 Conv_0_aggregated_features[1,0] + Conv_0_t[1,5]
   = 2.803231716156006
 Conv_0_output_constraint[1,6]: 0.5099456906318665 X[1,0]
   + 1.153847813606262 Conv_0_aggregated_features[1,0] + Conv_0_t[1,6]
   = 2.349466323852539
 Conv_0_output_constraint[1,7]: 1.624484419822693 X[1,0]
   + 0.069025494158268 Conv_0_aggregated_features[1,0] + Conv_0_t[1,7]
   = 0.9157499670982361
 Conv_0_output_constraint[1,8]: 0.8975760936737061 X[1,0]
   + 0.0988729 Conv_0_aggregated_features[1,0] + Conv_0_t[1,8]
   = -0.059127077460289
 Conv_0_output_constraint[1,9]: 1.157573580741882 X[1,0]
   + 0.00635915342718363 Conv_0_aggregated_features[1,0] + Conv_0_t[1,9]
   = -1.259021520614624
 Conv_0_output_constraint[1,10]: 1.51606810092926 X[1,0]
   + 0.1159021332859993 Conv_0_aggregated_features[1,0] + Conv_0_t[1,10]
   = -1.292696356773376
 Conv_0_output_constraint[1,11]: - 0.3291687667369843 X[1,0]
   + 1.914036870002747 Conv_0_aggregated_features[1,0] + Conv_0_t[1,11]
   = 1.365784168243408
 Conv_0_output_constraint[1,12]: - 0.3346672058105469 X[1,0]
   + 0.311100959777832 Conv_0_aggregated_features[1,0] + Conv_0_t[1,12]
   = -3.282561302185059
 Conv_0_output_constraint[1,13]: 0.9488325715065002 X[1,0]
   + 0.2108359783887863 Conv_0_aggregated_features[1,0] + Conv_0_t[1,13]
   = 3.006435632705688
 Conv_0_output_constraint[1,14]: 1.274890184402466 X[1,0]
   - 0.7632738351821899 Conv_0_aggregated_features[1,0] + Conv_0_t[1,14]
   = -2.962445259094238
 Conv_0_output_constraint[1,15]: 0.0870685502886772 X[1,0]
   + 0.1614511907100677 Conv_0_aggregated_features[1,0] + Conv_0_t[1,15]
   = -1.766824722290039
 Conv_0_output_constraint[2,0]: 0.2364515662193298 X[2,0]
   - 0.00366021576337516 Conv_0_aggregated_features[2,0] + Conv_0_t[2,0]
   = -0.8696585297584534
 Conv_0_output_constraint[2,1]: - 1.855180263519287 X[2,0]
   + 1.083907604217529 Conv_0_aggregated_features[2,0] + Conv_0_t[2,1]
   = -0.9598205089569092
 Conv_0_output_constraint[2,2]: 1.485349774360657 X[2,0]
   + 0.046770416200161 Conv_0_aggregated_features[2,0] + Conv_0_t[2,2]
   = -1.321228504180908
 Conv_0_output_constraint[2,3]: 0.7752511501312256 X[2,0]
   + 0.6645643711090088 Conv_0_aggregated_features[2,0] + Conv_0_t[2,3]
   = -0.4734632968902588
 Conv_0_output_constraint[2,4]: 1.752691864967346 X[2,0]
   + 0.0825358927249908 Conv_0_aggregated_features[2,0] + Conv_0_t[2,4]
   = 1.639361500740051
 Conv_0_output_constraint[2,5]: 1.845853567123413 X[2,0]
   - 0.1997903436422348 Conv_0_aggregated_features[2,0] + Conv_0_t[2,5]
   = 2.803231716156006
 Conv_0_output_constraint[2,6]: 0.5099456906318665 X[2,0]
   + 1.153847813606262 Conv_0_aggregated_features[2,0] + Conv_0_t[2,6]
   = 2.349466323852539
 Conv_0_output_constraint[2,7]: 1.624484419822693 X[2,0]
   + 0.069025494158268 Conv_0_aggregated_features[2,0] + Conv_0_t[2,7]
   = 0.9157499670982361
 Conv_0_output_constraint[2,8]: 0.8975760936737061 X[2,0]
   + 0.0988729 Conv_0_aggregated_features[2,0] + Conv_0_t[2,8]
   = -0.059127077460289
 Conv_0_output_constraint[2,9]: 1.157573580741882 X[2,0]
   + 0.00635915342718363 Conv_0_aggregated_features[2,0] + Conv_0_t[2,9]
   = -1.259021520614624
 Conv_0_output_constraint[2,10]: 1.51606810092926 X[2,0]
   + 0.1159021332859993 Conv_0_aggregated_features[2,0] + Conv_0_t[2,10]
   = -1.292696356773376
 Conv_0_output_constraint[2,11]: - 0.3291687667369843 X[2,0]
   + 1.914036870002747 Conv_0_aggregated_features[2,0] + Conv_0_t[2,11]
   = 1.365784168243408
 Conv_0_output_constraint[2,12]: - 0.3346672058105469 X[2,0]
   + 0.311100959777832 Conv_0_aggregated_features[2,0] + Conv_0_t[2,12]
   = -3.282561302185059
 Conv_0_output_constraint[2,13]: 0.9488325715065002 X[2,0]
   + 0.2108359783887863 Conv_0_aggregated_features[2,0] + Conv_0_t[2,13]
   = 3.006435632705688
 Conv_0_output_constraint[2,14]: 1.274890184402466 X[2,0]
   - 0.7632738351821899 Conv_0_aggregated_features[2,0] + Conv_0_t[2,14]
   = -2.962445259094238
 Conv_0_output_constraint[2,15]: 0.0870685502886772 X[2,0]
   + 0.1614511907100677 Conv_0_aggregated_features[2,0] + Conv_0_t[2,15]
   = -1.766824722290039
 Conv_0_output_constraint[3,0]: 0.2364515662193298 X[3,0]
   - 0.00366021576337516 Conv_0_aggregated_features[3,0] + Conv_0_t[3,0]
   = -0.8696585297584534
 Conv_0_output_constraint[3,1]: - 1.855180263519287 X[3,0]
   + 1.083907604217529 Conv_0_aggregated_features[3,0] + Conv_0_t[3,1]
   = -0.9598205089569092
 Conv_0_output_constraint[3,2]: 1.485349774360657 X[3,0]
   + 0.046770416200161 Conv_0_aggregated_features[3,0] + Conv_0_t[3,2]
   = -1.321228504180908
 Conv_0_output_constraint[3,3]: 0.7752511501312256 X[3,0]
   + 0.6645643711090088 Conv_0_aggregated_features[3,0] + Conv_0_t[3,3]
   = -0.4734632968902588
 Conv_0_output_constraint[3,4]: 1.752691864967346 X[3,0]
   + 0.0825358927249908 Conv_0_aggregated_features[3,0] + Conv_0_t[3,4]
   = 1.639361500740051
 Conv_0_output_constraint[3,5]: 1.845853567123413 X[3,0]
   - 0.1997903436422348 Conv_0_aggregated_features[3,0] + Conv_0_t[3,5]
   = 2.803231716156006
 Conv_0_output_constraint[3,6]: 0.5099456906318665 X[3,0]
   + 1.153847813606262 Conv_0_aggregated_features[3,0] + Conv_0_t[3,6]
   = 2.349466323852539
 Conv_0_output_constraint[3,7]: 1.624484419822693 X[3,0]
   + 0.069025494158268 Conv_0_aggregated_features[3,0] + Conv_0_t[3,7]
   = 0.9157499670982361
 Conv_0_output_constraint[3,8]: 0.8975760936737061 X[3,0]
   + 0.0988729 Conv_0_aggregated_features[3,0] + Conv_0_t[3,8]
   = -0.059127077460289
 Conv_0_output_constraint[3,9]: 1.157573580741882 X[3,0]
   + 0.00635915342718363 Conv_0_aggregated_features[3,0] + Conv_0_t[3,9]
   = -1.259021520614624
 Conv_0_output_constraint[3,10]: 1.51606810092926 X[3,0]
   + 0.1159021332859993 Conv_0_aggregated_features[3,0] + Conv_0_t[3,10]
   = -1.292696356773376
 Conv_0_output_constraint[3,11]: - 0.3291687667369843 X[3,0]
   + 1.914036870002747 Conv_0_aggregated_features[3,0] + Conv_0_t[3,11]
   = 1.365784168243408
 Conv_0_output_constraint[3,12]: - 0.3346672058105469 X[3,0]
   + 0.311100959777832 Conv_0_aggregated_features[3,0] + Conv_0_t[3,12]
   = -3.282561302185059
 Conv_0_output_constraint[3,13]: 0.9488325715065002 X[3,0]
   + 0.2108359783887863 Conv_0_aggregated_features[3,0] + Conv_0_t[3,13]
   = 3.006435632705688
 Conv_0_output_constraint[3,14]: 1.274890184402466 X[3,0]
   - 0.7632738351821899 Conv_0_aggregated_features[3,0] + Conv_0_t[3,14]
   = -2.962445259094238
 Conv_0_output_constraint[3,15]: 0.0870685502886772 X[3,0]
   + 0.1614511907100677 Conv_0_aggregated_features[3,0] + Conv_0_t[3,15]
   = -1.766824722290039
 Conv_0_output_constraint[4,0]: 0.2364515662193298 X[4,0]
   - 0.00366021576337516 Conv_0_aggregated_features[4,0] + Conv_0_t[4,0]
   = -0.8696585297584534
 Conv_0_output_constraint[4,1]: - 1.855180263519287 X[4,0]
   + 1.083907604217529 Conv_0_aggregated_features[4,0] + Conv_0_t[4,1]
   = -0.9598205089569092
 Conv_0_output_constraint[4,2]: 1.485349774360657 X[4,0]
   + 0.046770416200161 Conv_0_aggregated_features[4,0] + Conv_0_t[4,2]
   = -1.321228504180908
 Conv_0_output_constraint[4,3]: 0.7752511501312256 X[4,0]
   + 0.6645643711090088 Conv_0_aggregated_features[4,0] + Conv_0_t[4,3]
   = -0.4734632968902588
 Conv_0_output_constraint[4,4]: 1.752691864967346 X[4,0]
   + 0.0825358927249908 Conv_0_aggregated_features[4,0] + Conv_0_t[4,4]
   = 1.639361500740051
 Conv_0_output_constraint[4,5]: 1.845853567123413 X[4,0]
   - 0.1997903436422348 Conv_0_aggregated_features[4,0] + Conv_0_t[4,5]
   = 2.803231716156006
 Conv_0_output_constraint[4,6]: 0.5099456906318665 X[4,0]
   + 1.153847813606262 Conv_0_aggregated_features[4,0] + Conv_0_t[4,6]
   = 2.349466323852539
 Conv_0_output_constraint[4,7]: 1.624484419822693 X[4,0]
   + 0.069025494158268 Conv_0_aggregated_features[4,0] + Conv_0_t[4,7]
   = 0.9157499670982361
 Conv_0_output_constraint[4,8]: 0.8975760936737061 X[4,0]
   + 0.0988729 Conv_0_aggregated_features[4,0] + Conv_0_t[4,8]
   = -0.059127077460289
 Conv_0_output_constraint[4,9]: 1.157573580741882 X[4,0]
   + 0.00635915342718363 Conv_0_aggregated_features[4,0] + Conv_0_t[4,9]
   = -1.259021520614624
 Conv_0_output_constraint[4,10]: 1.51606810092926 X[4,0]
   + 0.1159021332859993 Conv_0_aggregated_features[4,0] + Conv_0_t[4,10]
   = -1.292696356773376
 Conv_0_output_constraint[4,11]: - 0.3291687667369843 X[4,0]
   + 1.914036870002747 Conv_0_aggregated_features[4,0] + Conv_0_t[4,11]
   = 1.365784168243408
 Conv_0_output_constraint[4,12]: - 0.3346672058105469 X[4,0]
   + 0.311100959777832 Conv_0_aggregated_features[4,0] + Conv_0_t[4,12]
   = -3.282561302185059
 Conv_0_output_constraint[4,13]: 0.9488325715065002 X[4,0]
   + 0.2108359783887863 Conv_0_aggregated_features[4,0] + Conv_0_t[4,13]
   = 3.006435632705688
 Conv_0_output_constraint[4,14]: 1.274890184402466 X[4,0]
   - 0.7632738351821899 Conv_0_aggregated_features[4,0] + Conv_0_t[4,14]
   = -2.962445259094238
 Conv_0_output_constraint[4,15]: 0.0870685502886772 X[4,0]
   + 0.1614511907100677 Conv_0_aggregated_features[4,0] + Conv_0_t[4,15]
   = -1.766824722290039
 Conv_0_output_constraint[5,0]: 0.2364515662193298 X[5,0]
   - 0.00366021576337516 Conv_0_aggregated_features[5,0] + Conv_0_t[5,0]
   = -0.8696585297584534
 Conv_0_output_constraint[5,1]: - 1.855180263519287 X[5,0]
   + 1.083907604217529 Conv_0_aggregated_features[5,0] + Conv_0_t[5,1]
   = -0.9598205089569092
 Conv_0_output_constraint[5,2]: 1.485349774360657 X[5,0]
   + 0.046770416200161 Conv_0_aggregated_features[5,0] + Conv_0_t[5,2]
   = -1.321228504180908
 Conv_0_output_constraint[5,3]: 0.7752511501312256 X[5,0]
   + 0.6645643711090088 Conv_0_aggregated_features[5,0] + Conv_0_t[5,3]
   = -0.4734632968902588
 Conv_0_output_constraint[5,4]: 1.752691864967346 X[5,0]
   + 0.0825358927249908 Conv_0_aggregated_features[5,0] + Conv_0_t[5,4]
   = 1.639361500740051
 Conv_0_output_constraint[5,5]: 1.845853567123413 X[5,0]
   - 0.1997903436422348 Conv_0_aggregated_features[5,0] + Conv_0_t[5,5]
   = 2.803231716156006
 Conv_0_output_constraint[5,6]: 0.5099456906318665 X[5,0]
   + 1.153847813606262 Conv_0_aggregated_features[5,0] + Conv_0_t[5,6]
   = 2.349466323852539
 Conv_0_output_constraint[5,7]: 1.624484419822693 X[5,0]
   + 0.069025494158268 Conv_0_aggregated_features[5,0] + Conv_0_t[5,7]
   = 0.9157499670982361
 Conv_0_output_constraint[5,8]: 0.8975760936737061 X[5,0]
   + 0.0988729 Conv_0_aggregated_features[5,0] + Conv_0_t[5,8]
   = -0.059127077460289
 Conv_0_output_constraint[5,9]: 1.157573580741882 X[5,0]
   + 0.00635915342718363 Conv_0_aggregated_features[5,0] + Conv_0_t[5,9]
   = -1.259021520614624
 Conv_0_output_constraint[5,10]: 1.51606810092926 X[5,0]
   + 0.1159021332859993 Conv_0_aggregated_features[5,0] + Conv_0_t[5,10]
   = -1.292696356773376
 Conv_0_output_constraint[5,11]: - 0.3291687667369843 X[5,0]
   + 1.914036870002747 Conv_0_aggregated_features[5,0] + Conv_0_t[5,11]
   = 1.365784168243408
 Conv_0_output_constraint[5,12]: - 0.3346672058105469 X[5,0]
   + 0.311100959777832 Conv_0_aggregated_features[5,0] + Conv_0_t[5,12]
   = -3.282561302185059
 Conv_0_output_constraint[5,13]: 0.9488325715065002 X[5,0]
   + 0.2108359783887863 Conv_0_aggregated_features[5,0] + Conv_0_t[5,13]
   = 3.006435632705688
 Conv_0_output_constraint[5,14]: 1.274890184402466 X[5,0]
   - 0.7632738351821899 Conv_0_aggregated_features[5,0] + Conv_0_t[5,14]
   = -2.962445259094238
 Conv_0_output_constraint[5,15]: 0.0870685502886772 X[5,0]
   + 0.1614511907100677 Conv_0_aggregated_features[5,0] + Conv_0_t[5,15]
   = -1.766824722290039
 Conv_0_output_constraint[6,0]: 0.2364515662193298 X[6,0]
   - 0.00366021576337516 Conv_0_aggregated_features[6,0] + Conv_0_t[6,0]
   = -0.8696585297584534
 Conv_0_output_constraint[6,1]: - 1.855180263519287 X[6,0]
   + 1.083907604217529 Conv_0_aggregated_features[6,0] + Conv_0_t[6,1]
   = -0.9598205089569092
 Conv_0_output_constraint[6,2]: 1.485349774360657 X[6,0]
   + 0.046770416200161 Conv_0_aggregated_features[6,0] + Conv_0_t[6,2]
   = -1.321228504180908
 Conv_0_output_constraint[6,3]: 0.7752511501312256 X[6,0]
   + 0.6645643711090088 Conv_0_aggregated_features[6,0] + Conv_0_t[6,3]
   = -0.4734632968902588
 Conv_0_output_constraint[6,4]: 1.752691864967346 X[6,0]
   + 0.0825358927249908 Conv_0_aggregated_features[6,0] + Conv_0_t[6,4]
   = 1.639361500740051
 Conv_0_output_constraint[6,5]: 1.845853567123413 X[6,0]
   - 0.1997903436422348 Conv_0_aggregated_features[6,0] + Conv_0_t[6,5]
   = 2.803231716156006
 Conv_0_output_constraint[6,6]: 0.5099456906318665 X[6,0]
   + 1.153847813606262 Conv_0_aggregated_features[6,0] + Conv_0_t[6,6]
   = 2.349466323852539
 Conv_0_output_constraint[6,7]: 1.624484419822693 X[6,0]
   + 0.069025494158268 Conv_0_aggregated_features[6,0] + Conv_0_t[6,7]
   = 0.9157499670982361
 Conv_0_output_constraint[6,8]: 0.8975760936737061 X[6,0]
   + 0.0988729 Conv_0_aggregated_features[6,0] + Conv_0_t[6,8]
   = -0.059127077460289
 Conv_0_output_constraint[6,9]: 1.157573580741882 X[6,0]
   + 0.00635915342718363 Conv_0_aggregated_features[6,0] + Conv_0_t[6,9]
   = -1.259021520614624
 Conv_0_output_constraint[6,10]: 1.51606810092926 X[6,0]
   + 0.1159021332859993 Conv_0_aggregated_features[6,0] + Conv_0_t[6,10]
   = -1.292696356773376
 Conv_0_output_constraint[6,11]: - 0.3291687667369843 X[6,0]
   + 1.914036870002747 Conv_0_aggregated_features[6,0] + Conv_0_t[6,11]
   = 1.365784168243408
 Conv_0_output_constraint[6,12]: - 0.3346672058105469 X[6,0]
   + 0.311100959777832 Conv_0_aggregated_features[6,0] + Conv_0_t[6,12]
   = -3.282561302185059
 Conv_0_output_constraint[6,13]: 0.9488325715065002 X[6,0]
   + 0.2108359783887863 Conv_0_aggregated_features[6,0] + Conv_0_t[6,13]
   = 3.006435632705688
 Conv_0_output_constraint[6,14]: 1.274890184402466 X[6,0]
   - 0.7632738351821899 Conv_0_aggregated_features[6,0] + Conv_0_t[6,14]
   = -2.962445259094238
 Conv_0_output_constraint[6,15]: 0.0870685502886772 X[6,0]
   + 0.1614511907100677 Conv_0_aggregated_features[6,0] + Conv_0_t[6,15]
   = -1.766824722290039
 Conv_0_output_constraint[7,0]: 0.2364515662193298 X[7,0]
   - 0.00366021576337516 Conv_0_aggregated_features[7,0] + Conv_0_t[7,0]
   = -0.8696585297584534
 Conv_0_output_constraint[7,1]: - 1.855180263519287 X[7,0]
   + 1.083907604217529 Conv_0_aggregated_features[7,0] + Conv_0_t[7,1]
   = -0.9598205089569092
 Conv_0_output_constraint[7,2]: 1.485349774360657 X[7,0]
   + 0.046770416200161 Conv_0_aggregated_features[7,0] + Conv_0_t[7,2]
   = -1.321228504180908
 Conv_0_output_constraint[7,3]: 0.7752511501312256 X[7,0]
   + 0.6645643711090088 Conv_0_aggregated_features[7,0] + Conv_0_t[7,3]
   = -0.4734632968902588
 Conv_0_output_constraint[7,4]: 1.752691864967346 X[7,0]
   + 0.0825358927249908 Conv_0_aggregated_features[7,0] + Conv_0_t[7,4]
   = 1.639361500740051
 Conv_0_output_constraint[7,5]: 1.845853567123413 X[7,0]
   - 0.1997903436422348 Conv_0_aggregated_features[7,0] + Conv_0_t[7,5]
   = 2.803231716156006
 Conv_0_output_constraint[7,6]: 0.5099456906318665 X[7,0]
   + 1.153847813606262 Conv_0_aggregated_features[7,0] + Conv_0_t[7,6]
   = 2.349466323852539
 Conv_0_output_constraint[7,7]: 1.624484419822693 X[7,0]
   + 0.069025494158268 Conv_0_aggregated_features[7,0] + Conv_0_t[7,7]
   = 0.9157499670982361
 Conv_0_output_constraint[7,8]: 0.8975760936737061 X[7,0]
   + 0.0988729 Conv_0_aggregated_features[7,0] + Conv_0_t[7,8]
   = -0.059127077460289
 Conv_0_output_constraint[7,9]: 1.157573580741882 X[7,0]
   + 0.00635915342718363 Conv_0_aggregated_features[7,0] + Conv_0_t[7,9]
   = -1.259021520614624
 Conv_0_output_constraint[7,10]: 1.51606810092926 X[7,0]
   + 0.1159021332859993 Conv_0_aggregated_features[7,0] + Conv_0_t[7,10]
   = -1.292696356773376
 Conv_0_output_constraint[7,11]: - 0.3291687667369843 X[7,0]
   + 1.914036870002747 Conv_0_aggregated_features[7,0] + Conv_0_t[7,11]
   = 1.365784168243408
 Conv_0_output_constraint[7,12]: - 0.3346672058105469 X[7,0]
   + 0.311100959777832 Conv_0_aggregated_features[7,0] + Conv_0_t[7,12]
   = -3.282561302185059
 Conv_0_output_constraint[7,13]: 0.9488325715065002 X[7,0]
   + 0.2108359783887863 Conv_0_aggregated_features[7,0] + Conv_0_t[7,13]
   = 3.006435632705688
 Conv_0_output_constraint[7,14]: 1.274890184402466 X[7,0]
   - 0.7632738351821899 Conv_0_aggregated_features[7,0] + Conv_0_t[7,14]
   = -2.962445259094238
 Conv_0_output_constraint[7,15]: 0.0870685502886772 X[7,0]
   + 0.1614511907100677 Conv_0_aggregated_features[7,0] + Conv_0_t[7,15]
   = -1.766824722290039
 Conv_0_output_constraint[8,0]: 0.2364515662193298 X[8,0]
   - 0.00366021576337516 Conv_0_aggregated_features[8,0] + Conv_0_t[8,0]
   = -0.8696585297584534
 Conv_0_output_constraint[8,1]: - 1.855180263519287 X[8,0]
   + 1.083907604217529 Conv_0_aggregated_features[8,0] + Conv_0_t[8,1]
   = -0.9598205089569092
 Conv_0_output_constraint[8,2]: 1.485349774360657 X[8,0]
   + 0.046770416200161 Conv_0_aggregated_features[8,0] + Conv_0_t[8,2]
   = -1.321228504180908
 Conv_0_output_constraint[8,3]: 0.7752511501312256 X[8,0]
   + 0.6645643711090088 Conv_0_aggregated_features[8,0] + Conv_0_t[8,3]
   = -0.4734632968902588
 Conv_0_output_constraint[8,4]: 1.752691864967346 X[8,0]
   + 0.0825358927249908 Conv_0_aggregated_features[8,0] + Conv_0_t[8,4]
   = 1.639361500740051
 Conv_0_output_constraint[8,5]: 1.845853567123413 X[8,0]
   - 0.1997903436422348 Conv_0_aggregated_features[8,0] + Conv_0_t[8,5]
   = 2.803231716156006
 Conv_0_output_constraint[8,6]: 0.5099456906318665 X[8,0]
   + 1.153847813606262 Conv_0_aggregated_features[8,0] + Conv_0_t[8,6]
   = 2.349466323852539
 Conv_0_output_constraint[8,7]: 1.624484419822693 X[8,0]
   + 0.069025494158268 Conv_0_aggregated_features[8,0] + Conv_0_t[8,7]
   = 0.9157499670982361
 Conv_0_output_constraint[8,8]: 0.8975760936737061 X[8,0]
   + 0.0988729 Conv_0_aggregated_features[8,0] + Conv_0_t[8,8]
   = -0.059127077460289
 Conv_0_output_constraint[8,9]: 1.157573580741882 X[8,0]
   + 0.00635915342718363 Conv_0_aggregated_features[8,0] + Conv_0_t[8,9]
   = -1.259021520614624
 Conv_0_output_constraint[8,10]: 1.51606810092926 X[8,0]
   + 0.1159021332859993 Conv_0_aggregated_features[8,0] + Conv_0_t[8,10]
   = -1.292696356773376
 Conv_0_output_constraint[8,11]: - 0.3291687667369843 X[8,0]
   + 1.914036870002747 Conv_0_aggregated_features[8,0] + Conv_0_t[8,11]
   = 1.365784168243408
 Conv_0_output_constraint[8,12]: - 0.3346672058105469 X[8,0]
   + 0.311100959777832 Conv_0_aggregated_features[8,0] + Conv_0_t[8,12]
   = -3.282561302185059
 Conv_0_output_constraint[8,13]: 0.9488325715065002 X[8,0]
   + 0.2108359783887863 Conv_0_aggregated_features[8,0] + Conv_0_t[8,13]
   = 3.006435632705688
 Conv_0_output_constraint[8,14]: 1.274890184402466 X[8,0]
   - 0.7632738351821899 Conv_0_aggregated_features[8,0] + Conv_0_t[8,14]
   = -2.962445259094238
 Conv_0_output_constraint[8,15]: 0.0870685502886772 X[8,0]
   + 0.1614511907100677 Conv_0_aggregated_features[8,0] + Conv_0_t[8,15]
   = -1.766824722290039
 Conv_0_output_constraint[9,0]: 0.2364515662193298 X[9,0]
   - 0.00366021576337516 Conv_0_aggregated_features[9,0] + Conv_0_t[9,0]
   = -0.8696585297584534
 Conv_0_output_constraint[9,1]: - 1.855180263519287 X[9,0]
   + 1.083907604217529 Conv_0_aggregated_features[9,0] + Conv_0_t[9,1]
   = -0.9598205089569092
 Conv_0_output_constraint[9,2]: 1.485349774360657 X[9,0]
   + 0.046770416200161 Conv_0_aggregated_features[9,0] + Conv_0_t[9,2]
   = -1.321228504180908
 Conv_0_output_constraint[9,3]: 0.7752511501312256 X[9,0]
   + 0.6645643711090088 Conv_0_aggregated_features[9,0] + Conv_0_t[9,3]
   = -0.4734632968902588
 Conv_0_output_constraint[9,4]: 1.752691864967346 X[9,0]
   + 0.0825358927249908 Conv_0_aggregated_features[9,0] + Conv_0_t[9,4]
   = 1.639361500740051
 Conv_0_output_constraint[9,5]: 1.845853567123413 X[9,0]
   - 0.1997903436422348 Conv_0_aggregated_features[9,0] + Conv_0_t[9,5]
   = 2.803231716156006
 Conv_0_output_constraint[9,6]: 0.5099456906318665 X[9,0]
   + 1.153847813606262 Conv_0_aggregated_features[9,0] + Conv_0_t[9,6]
   = 2.349466323852539
 Conv_0_output_constraint[9,7]: 1.624484419822693 X[9,0]
   + 0.069025494158268 Conv_0_aggregated_features[9,0] + Conv_0_t[9,7]
   = 0.9157499670982361
 Conv_0_output_constraint[9,8]: 0.8975760936737061 X[9,0]
   + 0.0988729 Conv_0_aggregated_features[9,0] + Conv_0_t[9,8]
   = -0.059127077460289
 Conv_0_output_constraint[9,9]: 1.157573580741882 X[9,0]
   + 0.00635915342718363 Conv_0_aggregated_features[9,0] + Conv_0_t[9,9]
   = -1.259021520614624
 Conv_0_output_constraint[9,10]: 1.51606810092926 X[9,0]
   + 0.1159021332859993 Conv_0_aggregated_features[9,0] + Conv_0_t[9,10]
   = -1.292696356773376
 Conv_0_output_constraint[9,11]: - 0.3291687667369843 X[9,0]
   + 1.914036870002747 Conv_0_aggregated_features[9,0] + Conv_0_t[9,11]
   = 1.365784168243408
 Conv_0_output_constraint[9,12]: - 0.3346672058105469 X[9,0]
   + 0.311100959777832 Conv_0_aggregated_features[9,0] + Conv_0_t[9,12]
   = -3.282561302185059
 Conv_0_output_constraint[9,13]: 0.9488325715065002 X[9,0]
   + 0.2108359783887863 Conv_0_aggregated_features[9,0] + Conv_0_t[9,13]
   = 3.006435632705688
 Conv_0_output_constraint[9,14]: 1.274890184402466 X[9,0]
   - 0.7632738351821899 Conv_0_aggregated_features[9,0] + Conv_0_t[9,14]
   = -2.962445259094238
 Conv_0_output_constraint[9,15]: 0.0870685502886772 X[9,0]
   + 0.1614511907100677 Conv_0_aggregated_features[9,0] + Conv_0_t[9,15]
   = -1.766824722290039
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
 Conv_0_Relu_constraint_1[8,0]: - Conv_0_t[8,0] + Conv_0_Relu_ts[8,0]
   - Conv_0_Relu_ss[8,0] = 0
 Conv_0_Relu_constraint_1[8,1]: - Conv_0_t[8,1] + Conv_0_Relu_ts[8,1]
   - Conv_0_Relu_ss[8,1] = 0
 Conv_0_Relu_constraint_1[8,2]: - Conv_0_t[8,2] + Conv_0_Relu_ts[8,2]
   - Conv_0_Relu_ss[8,2] = 0
 Conv_0_Relu_constraint_1[8,3]: - Conv_0_t[8,3] + Conv_0_Relu_ts[8,3]
   - Conv_0_Relu_ss[8,3] = 0
 Conv_0_Relu_constraint_1[8,4]: - Conv_0_t[8,4] + Conv_0_Relu_ts[8,4]
   - Conv_0_Relu_ss[8,4] = 0
 Conv_0_Relu_constraint_1[8,5]: - Conv_0_t[8,5] + Conv_0_Relu_ts[8,5]
   - Conv_0_Relu_ss[8,5] = 0
 Conv_0_Relu_constraint_1[8,6]: - Conv_0_t[8,6] + Conv_0_Relu_ts[8,6]
   - Conv_0_Relu_ss[8,6] = 0
 Conv_0_Relu_constraint_1[8,7]: - Conv_0_t[8,7] + Conv_0_Relu_ts[8,7]
   - Conv_0_Relu_ss[8,7] = 0
 Conv_0_Relu_constraint_1[8,8]: - Conv_0_t[8,8] + Conv_0_Relu_ts[8,8]
   - Conv_0_Relu_ss[8,8] = 0
 Conv_0_Relu_constraint_1[8,9]: - Conv_0_t[8,9] + Conv_0_Relu_ts[8,9]
   - Conv_0_Relu_ss[8,9] = 0
 Conv_0_Relu_constraint_1[8,10]: - Conv_0_t[8,10] + Conv_0_Relu_ts[8,10]
   - Conv_0_Relu_ss[8,10] = 0
 Conv_0_Relu_constraint_1[8,11]: - Conv_0_t[8,11] + Conv_0_Relu_ts[8,11]
   - Conv_0_Relu_ss[8,11] = 0
 Conv_0_Relu_constraint_1[8,12]: - Conv_0_t[8,12] + Conv_0_Relu_ts[8,12]
   - Conv_0_Relu_ss[8,12] = 0
 Conv_0_Relu_constraint_1[8,13]: - Conv_0_t[8,13] + Conv_0_Relu_ts[8,13]
   - Conv_0_Relu_ss[8,13] = 0
 Conv_0_Relu_constraint_1[8,14]: - Conv_0_t[8,14] + Conv_0_Relu_ts[8,14]
   - Conv_0_Relu_ss[8,14] = 0
 Conv_0_Relu_constraint_1[8,15]: - Conv_0_t[8,15] + Conv_0_Relu_ts[8,15]
   - Conv_0_Relu_ss[8,15] = 0
 Conv_0_Relu_constraint_1[9,0]: - Conv_0_t[9,0] + Conv_0_Relu_ts[9,0]
   - Conv_0_Relu_ss[9,0] = 0
 Conv_0_Relu_constraint_1[9,1]: - Conv_0_t[9,1] + Conv_0_Relu_ts[9,1]
   - Conv_0_Relu_ss[9,1] = 0
 Conv_0_Relu_constraint_1[9,2]: - Conv_0_t[9,2] + Conv_0_Relu_ts[9,2]
   - Conv_0_Relu_ss[9,2] = 0
 Conv_0_Relu_constraint_1[9,3]: - Conv_0_t[9,3] + Conv_0_Relu_ts[9,3]
   - Conv_0_Relu_ss[9,3] = 0
 Conv_0_Relu_constraint_1[9,4]: - Conv_0_t[9,4] + Conv_0_Relu_ts[9,4]
   - Conv_0_Relu_ss[9,4] = 0
 Conv_0_Relu_constraint_1[9,5]: - Conv_0_t[9,5] + Conv_0_Relu_ts[9,5]
   - Conv_0_Relu_ss[9,5] = 0
 Conv_0_Relu_constraint_1[9,6]: - Conv_0_t[9,6] + Conv_0_Relu_ts[9,6]
   - Conv_0_Relu_ss[9,6] = 0
 Conv_0_Relu_constraint_1[9,7]: - Conv_0_t[9,7] + Conv_0_Relu_ts[9,7]
   - Conv_0_Relu_ss[9,7] = 0
 Conv_0_Relu_constraint_1[9,8]: - Conv_0_t[9,8] + Conv_0_Relu_ts[9,8]
   - Conv_0_Relu_ss[9,8] = 0
 Conv_0_Relu_constraint_1[9,9]: - Conv_0_t[9,9] + Conv_0_Relu_ts[9,9]
   - Conv_0_Relu_ss[9,9] = 0
 Conv_0_Relu_constraint_1[9,10]: - Conv_0_t[9,10] + Conv_0_Relu_ts[9,10]
   - Conv_0_Relu_ss[9,10] = 0
 Conv_0_Relu_constraint_1[9,11]: - Conv_0_t[9,11] + Conv_0_Relu_ts[9,11]
   - Conv_0_Relu_ss[9,11] = 0
 Conv_0_Relu_constraint_1[9,12]: - Conv_0_t[9,12] + Conv_0_Relu_ts[9,12]
   - Conv_0_Relu_ss[9,12] = 0
 Conv_0_Relu_constraint_1[9,13]: - Conv_0_t[9,13] + Conv_0_Relu_ts[9,13]
   - Conv_0_Relu_ss[9,13] = 0
 Conv_0_Relu_constraint_1[9,14]: - Conv_0_t[9,14] + Conv_0_Relu_ts[9,14]
   - Conv_0_Relu_ss[9,14] = 0
 Conv_0_Relu_constraint_1[9,15]: - Conv_0_t[9,15] + Conv_0_Relu_ts[9,15]
   - Conv_0_Relu_ss[9,15] = 0
 Conv_0_Relu_constraint_2[0,0]: Conv_0_Relu_ts[0,0] <= 0
 Conv_0_Relu_constraint_2[0,1]: Conv_0_Relu_ts[0,1]
   - 17.59198212623596 C601 <= 0
 Conv_0_Relu_constraint_2[0,2]: Conv_0_Relu_ts[0,2] <= 0
 Conv_0_Relu_constraint_2[0,3]: Conv_0_Relu_ts[0,3] <= 0
 Conv_0_Relu_constraint_2[0,4]: Conv_0_Relu_ts[0,4]
   - 1.639361500740051 C604 <= 0
 Conv_0_Relu_constraint_2[0,5]: Conv_0_Relu_ts[0,5]
   - 4.801135152578354 C605 <= 0
 Conv_0_Relu_constraint_2[0,6]: Conv_0_Relu_ts[0,6]
   - 2.349466323852539 C606 <= 0
 Conv_0_Relu_constraint_2[0,7]: Conv_0_Relu_ts[0,7]
   - 0.9157499670982361 C607 <= 0
 Conv_0_Relu_constraint_2[0,8]: Conv_0_Relu_ts[0,8] <= 0
 Conv_0_Relu_constraint_2[0,9]: Conv_0_Relu_ts[0,9] <= 0
 Conv_0_Relu_constraint_2[0,10]: Conv_0_Relu_ts[0,10] <= 0
 Conv_0_Relu_constraint_2[0,11]: Conv_0_Relu_ts[0,11]
   - 4.657471835613251 C611 <= 0
 Conv_0_Relu_constraint_2[0,12]: Conv_0_Relu_ts[0,12]
   - 0.0641107559204102 C612 <= 0
 Conv_0_Relu_constraint_2[0,13]: Conv_0_Relu_ts[0,13]
   - 3.006435632705688 C613 <= 0
 Conv_0_Relu_constraint_2[0,14]: Conv_0_Relu_ts[0,14]
   - 4.670293092727661 C614 <= 0
 Conv_0_Relu_constraint_2[0,15]: Conv_0_Relu_ts[0,15] <= 0
 Conv_0_Relu_constraint_2[1,0]: Conv_0_Relu_ts[1,0] <= 0
 Conv_0_Relu_constraint_2[1,1]: Conv_0_Relu_ts[1,1]
   - 17.59198212623596 C617 <= 0
 Conv_0_Relu_constraint_2[1,2]: Conv_0_Relu_ts[1,2] <= 0
 Conv_0_Relu_constraint_2[1,3]: Conv_0_Relu_ts[1,3] <= 0
 Conv_0_Relu_constraint_2[1,4]: Conv_0_Relu_ts[1,4]
   - 1.639361500740051 C620 <= 0
 Conv_0_Relu_constraint_2[1,5]: Conv_0_Relu_ts[1,5]
   - 4.801135152578354 C621 <= 0
 Conv_0_Relu_constraint_2[1,6]: Conv_0_Relu_ts[1,6]
   - 2.349466323852539 C622 <= 0
 Conv_0_Relu_constraint_2[1,7]: Conv_0_Relu_ts[1,7]
   - 0.9157499670982361 C623 <= 0
 Conv_0_Relu_constraint_2[1,8]: Conv_0_Relu_ts[1,8] <= 0
 Conv_0_Relu_constraint_2[1,9]: Conv_0_Relu_ts[1,9] <= 0
 Conv_0_Relu_constraint_2[1,10]: Conv_0_Relu_ts[1,10] <= 0
 Conv_0_Relu_constraint_2[1,11]: Conv_0_Relu_ts[1,11]
   - 4.657471835613251 C627 <= 0
 Conv_0_Relu_constraint_2[1,12]: Conv_0_Relu_ts[1,12]
   - 0.0641107559204102 C628 <= 0
 Conv_0_Relu_constraint_2[1,13]: Conv_0_Relu_ts[1,13]
   - 3.006435632705688 C629 <= 0
 Conv_0_Relu_constraint_2[1,14]: Conv_0_Relu_ts[1,14]
   - 4.670293092727661 C630 <= 0
 Conv_0_Relu_constraint_2[1,15]: Conv_0_Relu_ts[1,15] <= 0
 Conv_0_Relu_constraint_2[2,0]: Conv_0_Relu_ts[2,0] <= 0
 Conv_0_Relu_constraint_2[2,1]: Conv_0_Relu_ts[2,1]
   - 17.59198212623596 C633 <= 0
 Conv_0_Relu_constraint_2[2,2]: Conv_0_Relu_ts[2,2] <= 0
 Conv_0_Relu_constraint_2[2,3]: Conv_0_Relu_ts[2,3] <= 0
 Conv_0_Relu_constraint_2[2,4]: Conv_0_Relu_ts[2,4]
   - 1.639361500740051 C636 <= 0
 Conv_0_Relu_constraint_2[2,5]: Conv_0_Relu_ts[2,5]
   - 4.801135152578354 C637 <= 0
 Conv_0_Relu_constraint_2[2,6]: Conv_0_Relu_ts[2,6]
   - 2.349466323852539 C638 <= 0
 Conv_0_Relu_constraint_2[2,7]: Conv_0_Relu_ts[2,7]
   - 0.9157499670982361 C639 <= 0
 Conv_0_Relu_constraint_2[2,8]: Conv_0_Relu_ts[2,8] <= 0
 Conv_0_Relu_constraint_2[2,9]: Conv_0_Relu_ts[2,9] <= 0
 Conv_0_Relu_constraint_2[2,10]: Conv_0_Relu_ts[2,10] <= 0
 Conv_0_Relu_constraint_2[2,11]: Conv_0_Relu_ts[2,11]
   - 4.657471835613251 C643 <= 0
 Conv_0_Relu_constraint_2[2,12]: Conv_0_Relu_ts[2,12]
   - 0.0641107559204102 C644 <= 0
 Conv_0_Relu_constraint_2[2,13]: Conv_0_Relu_ts[2,13]
   - 3.006435632705688 C645 <= 0
 Conv_0_Relu_constraint_2[2,14]: Conv_0_Relu_ts[2,14]
   - 4.670293092727661 C646 <= 0
 Conv_0_Relu_constraint_2[2,15]: Conv_0_Relu_ts[2,15] <= 0
 Conv_0_Relu_constraint_2[3,0]: Conv_0_Relu_ts[3,0] <= 0
 Conv_0_Relu_constraint_2[3,1]: Conv_0_Relu_ts[3,1]
   - 17.59198212623596 C649 <= 0
 Conv_0_Relu_constraint_2[3,2]: Conv_0_Relu_ts[3,2] <= 0
 Conv_0_Relu_constraint_2[3,3]: Conv_0_Relu_ts[3,3] <= 0
 Conv_0_Relu_constraint_2[3,4]: Conv_0_Relu_ts[3,4]
   - 1.639361500740051 C652 <= 0
 Conv_0_Relu_constraint_2[3,5]: Conv_0_Relu_ts[3,5]
   - 4.801135152578354 C653 <= 0
 Conv_0_Relu_constraint_2[3,6]: Conv_0_Relu_ts[3,6]
   - 2.349466323852539 C654 <= 0
 Conv_0_Relu_constraint_2[3,7]: Conv_0_Relu_ts[3,7]
   - 0.9157499670982361 C655 <= 0
 Conv_0_Relu_constraint_2[3,8]: Conv_0_Relu_ts[3,8] <= 0
 Conv_0_Relu_constraint_2[3,9]: Conv_0_Relu_ts[3,9] <= 0
 Conv_0_Relu_constraint_2[3,10]: Conv_0_Relu_ts[3,10] <= 0
 Conv_0_Relu_constraint_2[3,11]: Conv_0_Relu_ts[3,11]
   - 4.657471835613251 C659 <= 0
 Conv_0_Relu_constraint_2[3,12]: Conv_0_Relu_ts[3,12]
   - 0.0641107559204102 C660 <= 0
 Conv_0_Relu_constraint_2[3,13]: Conv_0_Relu_ts[3,13]
   - 3.006435632705688 C661 <= 0
 Conv_0_Relu_constraint_2[3,14]: Conv_0_Relu_ts[3,14]
   - 4.670293092727661 C662 <= 0
 Conv_0_Relu_constraint_2[3,15]: Conv_0_Relu_ts[3,15] <= 0
 Conv_0_Relu_constraint_2[4,0]: Conv_0_Relu_ts[4,0] <= 0
 Conv_0_Relu_constraint_2[4,1]: Conv_0_Relu_ts[4,1]
   - 17.59198212623596 C665 <= 0
 Conv_0_Relu_constraint_2[4,2]: Conv_0_Relu_ts[4,2] <= 0
 Conv_0_Relu_constraint_2[4,3]: Conv_0_Relu_ts[4,3] <= 0
 Conv_0_Relu_constraint_2[4,4]: Conv_0_Relu_ts[4,4]
   - 1.639361500740051 C668 <= 0
 Conv_0_Relu_constraint_2[4,5]: Conv_0_Relu_ts[4,5]
   - 4.801135152578354 C669 <= 0
 Conv_0_Relu_constraint_2[4,6]: Conv_0_Relu_ts[4,6]
   - 2.349466323852539 C670 <= 0
 Conv_0_Relu_constraint_2[4,7]: Conv_0_Relu_ts[4,7]
   - 0.9157499670982361 C671 <= 0
 Conv_0_Relu_constraint_2[4,8]: Conv_0_Relu_ts[4,8] <= 0
 Conv_0_Relu_constraint_2[4,9]: Conv_0_Relu_ts[4,9] <= 0
 Conv_0_Relu_constraint_2[4,10]: Conv_0_Relu_ts[4,10] <= 0
 Conv_0_Relu_constraint_2[4,11]: Conv_0_Relu_ts[4,11]
   - 4.657471835613251 C675 <= 0
 Conv_0_Relu_constraint_2[4,12]: Conv_0_Relu_ts[4,12]
   - 0.0641107559204102 C676 <= 0
 Conv_0_Relu_constraint_2[4,13]: Conv_0_Relu_ts[4,13]
   - 3.006435632705688 C677 <= 0
 Conv_0_Relu_constraint_2[4,14]: Conv_0_Relu_ts[4,14]
   - 4.670293092727661 C678 <= 0
 Conv_0_Relu_constraint_2[4,15]: Conv_0_Relu_ts[4,15] <= 0
 Conv_0_Relu_constraint_2[5,0]: Conv_0_Relu_ts[5,0] <= 0
 Conv_0_Relu_constraint_2[5,1]: Conv_0_Relu_ts[5,1]
   - 17.59198212623596 C681 <= 0
 Conv_0_Relu_constraint_2[5,2]: Conv_0_Relu_ts[5,2] <= 0
 Conv_0_Relu_constraint_2[5,3]: Conv_0_Relu_ts[5,3] <= 0
 Conv_0_Relu_constraint_2[5,4]: Conv_0_Relu_ts[5,4]
   - 1.639361500740051 C684 <= 0
 Conv_0_Relu_constraint_2[5,5]: Conv_0_Relu_ts[5,5]
   - 4.801135152578354 C685 <= 0
 Conv_0_Relu_constraint_2[5,6]: Conv_0_Relu_ts[5,6]
   - 2.349466323852539 C686 <= 0
 Conv_0_Relu_constraint_2[5,7]: Conv_0_Relu_ts[5,7]
   - 0.9157499670982361 C687 <= 0
 Conv_0_Relu_constraint_2[5,8]: Conv_0_Relu_ts[5,8] <= 0
 Conv_0_Relu_constraint_2[5,9]: Conv_0_Relu_ts[5,9] <= 0
 Conv_0_Relu_constraint_2[5,10]: Conv_0_Relu_ts[5,10] <= 0
 Conv_0_Relu_constraint_2[5,11]: Conv_0_Relu_ts[5,11]
   - 4.657471835613251 C691 <= 0
 Conv_0_Relu_constraint_2[5,12]: Conv_0_Relu_ts[5,12]
   - 0.0641107559204102 C692 <= 0
 Conv_0_Relu_constraint_2[5,13]: Conv_0_Relu_ts[5,13]
   - 3.006435632705688 C693 <= 0
 Conv_0_Relu_constraint_2[5,14]: Conv_0_Relu_ts[5,14]
   - 4.670293092727661 C694 <= 0
 Conv_0_Relu_constraint_2[5,15]: Conv_0_Relu_ts[5,15] <= 0
 Conv_0_Relu_constraint_2[6,0]: Conv_0_Relu_ts[6,0] <= 0
 Conv_0_Relu_constraint_2[6,1]: Conv_0_Relu_ts[6,1]
   - 17.59198212623596 C697 <= 0
 Conv_0_Relu_constraint_2[6,2]: Conv_0_Relu_ts[6,2] <= 0
 Conv_0_Relu_constraint_2[6,3]: Conv_0_Relu_ts[6,3] <= 0
 Conv_0_Relu_constraint_2[6,4]: Conv_0_Relu_ts[6,4]
   - 1.639361500740051 C700 <= 0
 Conv_0_Relu_constraint_2[6,5]: Conv_0_Relu_ts[6,5]
   - 4.801135152578354 C701 <= 0
 Conv_0_Relu_constraint_2[6,6]: Conv_0_Relu_ts[6,6]
   - 2.349466323852539 C702 <= 0
 Conv_0_Relu_constraint_2[6,7]: Conv_0_Relu_ts[6,7]
   - 0.9157499670982361 C703 <= 0
 Conv_0_Relu_constraint_2[6,8]: Conv_0_Relu_ts[6,8] <= 0
 Conv_0_Relu_constraint_2[6,9]: Conv_0_Relu_ts[6,9] <= 0
 Conv_0_Relu_constraint_2[6,10]: Conv_0_Relu_ts[6,10] <= 0
 Conv_0_Relu_constraint_2[6,11]: Conv_0_Relu_ts[6,11]
   - 4.657471835613251 C707 <= 0
 Conv_0_Relu_constraint_2[6,12]: Conv_0_Relu_ts[6,12]
   - 0.0641107559204102 C708 <= 0
 Conv_0_Relu_constraint_2[6,13]: Conv_0_Relu_ts[6,13]
   - 3.006435632705688 C709 <= 0
 Conv_0_Relu_constraint_2[6,14]: Conv_0_Relu_ts[6,14]
   - 4.670293092727661 C710 <= 0
 Conv_0_Relu_constraint_2[6,15]: Conv_0_Relu_ts[6,15] <= 0
 Conv_0_Relu_constraint_2[7,0]: Conv_0_Relu_ts[7,0] <= 0
 Conv_0_Relu_constraint_2[7,1]: Conv_0_Relu_ts[7,1]
   - 17.59198212623596 C713 <= 0
 Conv_0_Relu_constraint_2[7,2]: Conv_0_Relu_ts[7,2] <= 0
 Conv_0_Relu_constraint_2[7,3]: Conv_0_Relu_ts[7,3] <= 0
 Conv_0_Relu_constraint_2[7,4]: Conv_0_Relu_ts[7,4]
   - 1.639361500740051 C716 <= 0
 Conv_0_Relu_constraint_2[7,5]: Conv_0_Relu_ts[7,5]
   - 4.801135152578354 C717 <= 0
 Conv_0_Relu_constraint_2[7,6]: Conv_0_Relu_ts[7,6]
   - 2.349466323852539 C718 <= 0
 Conv_0_Relu_constraint_2[7,7]: Conv_0_Relu_ts[7,7]
   - 0.9157499670982361 C719 <= 0
 Conv_0_Relu_constraint_2[7,8]: Conv_0_Relu_ts[7,8] <= 0
 Conv_0_Relu_constraint_2[7,9]: Conv_0_Relu_ts[7,9] <= 0
 Conv_0_Relu_constraint_2[7,10]: Conv_0_Relu_ts[7,10] <= 0
 Conv_0_Relu_constraint_2[7,11]: Conv_0_Relu_ts[7,11]
   - 4.657471835613251 C723 <= 0
 Conv_0_Relu_constraint_2[7,12]: Conv_0_Relu_ts[7,12]
   - 0.0641107559204102 C724 <= 0
 Conv_0_Relu_constraint_2[7,13]: Conv_0_Relu_ts[7,13]
   - 3.006435632705688 C725 <= 0
 Conv_0_Relu_constraint_2[7,14]: Conv_0_Relu_ts[7,14]
   - 4.670293092727661 C726 <= 0
 Conv_0_Relu_constraint_2[7,15]: Conv_0_Relu_ts[7,15] <= 0
 Conv_0_Relu_constraint_2[8,0]: Conv_0_Relu_ts[8,0] <= 0
 Conv_0_Relu_constraint_2[8,1]: Conv_0_Relu_ts[8,1]
   - 17.59198212623596 C729 <= 0
 Conv_0_Relu_constraint_2[8,2]: Conv_0_Relu_ts[8,2] <= 0
 Conv_0_Relu_constraint_2[8,3]: Conv_0_Relu_ts[8,3] <= 0
 Conv_0_Relu_constraint_2[8,4]: Conv_0_Relu_ts[8,4]
   - 1.639361500740051 C732 <= 0
 Conv_0_Relu_constraint_2[8,5]: Conv_0_Relu_ts[8,5]
   - 4.801135152578354 C733 <= 0
 Conv_0_Relu_constraint_2[8,6]: Conv_0_Relu_ts[8,6]
   - 2.349466323852539 C734 <= 0
 Conv_0_Relu_constraint_2[8,7]: Conv_0_Relu_ts[8,7]
   - 0.9157499670982361 C735 <= 0
 Conv_0_Relu_constraint_2[8,8]: Conv_0_Relu_ts[8,8] <= 0
 Conv_0_Relu_constraint_2[8,9]: Conv_0_Relu_ts[8,9] <= 0
 Conv_0_Relu_constraint_2[8,10]: Conv_0_Relu_ts[8,10] <= 0
 Conv_0_Relu_constraint_2[8,11]: Conv_0_Relu_ts[8,11]
   - 4.657471835613251 C739 <= 0
 Conv_0_Relu_constraint_2[8,12]: Conv_0_Relu_ts[8,12]
   - 0.0641107559204102 C740 <= 0
 Conv_0_Relu_constraint_2[8,13]: Conv_0_Relu_ts[8,13]
   - 3.006435632705688 C741 <= 0
 Conv_0_Relu_constraint_2[8,14]: Conv_0_Relu_ts[8,14]
   - 4.670293092727661 C742 <= 0
 Conv_0_Relu_constraint_2[8,15]: Conv_0_Relu_ts[8,15] <= 0
 Conv_0_Relu_constraint_2[9,0]: Conv_0_Relu_ts[9,0] <= 0
 Conv_0_Relu_constraint_2[9,1]: Conv_0_Relu_ts[9,1]
   - 17.59198212623596 C745 <= 0
 Conv_0_Relu_constraint_2[9,2]: Conv_0_Relu_ts[9,2] <= 0
 Conv_0_Relu_constraint_2[9,3]: Conv_0_Relu_ts[9,3] <= 0
 Conv_0_Relu_constraint_2[9,4]: Conv_0_Relu_ts[9,4]
   - 1.639361500740051 C748 <= 0
 Conv_0_Relu_constraint_2[9,5]: Conv_0_Relu_ts[9,5]
   - 4.801135152578354 C749 <= 0
 Conv_0_Relu_constraint_2[9,6]: Conv_0_Relu_ts[9,6]
   - 2.349466323852539 C750 <= 0
 Conv_0_Relu_constraint_2[9,7]: Conv_0_Relu_ts[9,7]
   - 0.9157499670982361 C751 <= 0
 Conv_0_Relu_constraint_2[9,8]: Conv_0_Relu_ts[9,8] <= 0
 Conv_0_Relu_constraint_2[9,9]: Conv_0_Relu_ts[9,9] <= 0
 Conv_0_Relu_constraint_2[9,10]: Conv_0_Relu_ts[9,10] <= 0
 Conv_0_Relu_constraint_2[9,11]: Conv_0_Relu_ts[9,11]
   - 4.657471835613251 C755 <= 0
 Conv_0_Relu_constraint_2[9,12]: Conv_0_Relu_ts[9,12]
   - 0.0641107559204102 C756 <= 0
 Conv_0_Relu_constraint_2[9,13]: Conv_0_Relu_ts[9,13]
   - 3.006435632705688 C757 <= 0
 Conv_0_Relu_constraint_2[9,14]: Conv_0_Relu_ts[9,14]
   - 4.670293092727661 C758 <= 0
 Conv_0_Relu_constraint_2[9,15]: Conv_0_Relu_ts[9,15] <= 0
 Conv_0_Relu_constraint_3[0,0]: Conv_0_Relu_ss[0,0]
   + 3.234174191951752 C600 <= 3.234174191951752
 Conv_0_Relu_constraint_3[0,1]: Conv_0_Relu_ss[0,1] + 11.7988965511322 C601
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[0,2]: Conv_0_Relu_ss[0,2]
   + 16.64243040978909 C602 <= 16.64243040978909
 Conv_0_Relu_constraint_3[0,3]: Conv_0_Relu_ss[0,3] + 14.8716185092926 C603
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[0,4]: Conv_0_Relu_ss[0,4]
   + 16.71291607618332 C604 <= 16.71291607618332
 Conv_0_Relu_constraint_3[0,5]: Conv_0_Relu_ss[0,5]
   + 15.65530395507812 C605 <= 15.65530395507812
 Conv_0_Relu_constraint_3[0,6]: Conv_0_Relu_ss[0,6]
   + 14.28846871852875 C606 <= 14.28846871852875
 Conv_0_Relu_constraint_3[0,7]: Conv_0_Relu_ss[0,7]
   + 16.01934917271137 C607 <= 16.01934917271137
 Conv_0_Relu_constraint_3[0,8]: Conv_0_Relu_ss[0,8] + 10.0236170142889 C608
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[0,9]: Conv_0_Relu_ss[0,9]
   + 12.89834886230528 C609 <= 12.89834886230528
 Conv_0_Relu_constraint_3[0,10]: Conv_0_Relu_ss[0,10]
   + 17.61239869892597 C610 <= 17.61239869892597
 Conv_0_Relu_constraint_3[0,11]: Conv_0_Relu_ss[0,11]
   + 17.77458453178406 C611 <= 17.77458453178406
 Conv_0_Relu_constraint_3[0,12]: Conv_0_Relu_ss[0,12]
   + 6.393570899963379 C612 <= 6.393570899963379
 Conv_0_Relu_constraint_3[0,13]: Conv_0_Relu_ss[0,13]
   + 8.590249866247177 C613 <= 8.590249866247177
 Conv_0_Relu_constraint_3[0,14]: Conv_0_Relu_ss[0,14]
   + 15.7113471031189 C614 <= 15.7113471031189
 Conv_0_Relu_constraint_3[0,15]: Conv_0_Relu_ss[0,15]
   + 4.252022132277489 C615 <= 4.252022132277489
 Conv_0_Relu_constraint_3[1,0]: Conv_0_Relu_ss[1,0]
   + 3.234174191951752 C616 <= 3.234174191951752
 Conv_0_Relu_constraint_3[1,1]: Conv_0_Relu_ss[1,1] + 11.7988965511322 C617
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[1,2]: Conv_0_Relu_ss[1,2]
   + 16.64243040978909 C618 <= 16.64243040978909
 Conv_0_Relu_constraint_3[1,3]: Conv_0_Relu_ss[1,3] + 14.8716185092926 C619
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[1,4]: Conv_0_Relu_ss[1,4]
   + 16.71291607618332 C620 <= 16.71291607618332
 Conv_0_Relu_constraint_3[1,5]: Conv_0_Relu_ss[1,5]
   + 15.65530395507812 C621 <= 15.65530395507812
 Conv_0_Relu_constraint_3[1,6]: Conv_0_Relu_ss[1,6]
   + 14.28846871852875 C622 <= 14.28846871852875
 Conv_0_Relu_constraint_3[1,7]: Conv_0_Relu_ss[1,7]
   + 16.01934917271137 C623 <= 16.01934917271137
 Conv_0_Relu_constraint_3[1,8]: Conv_0_Relu_ss[1,8] + 10.0236170142889 C624
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[1,9]: Conv_0_Relu_ss[1,9]
   + 12.89834886230528 C625 <= 12.89834886230528
 Conv_0_Relu_constraint_3[1,10]: Conv_0_Relu_ss[1,10]
   + 17.61239869892597 C626 <= 17.61239869892597
 Conv_0_Relu_constraint_3[1,11]: Conv_0_Relu_ss[1,11]
   + 17.77458453178406 C627 <= 17.77458453178406
 Conv_0_Relu_constraint_3[1,12]: Conv_0_Relu_ss[1,12]
   + 6.393570899963379 C628 <= 6.393570899963379
 Conv_0_Relu_constraint_3[1,13]: Conv_0_Relu_ss[1,13]
   + 8.590249866247177 C629 <= 8.590249866247177
 Conv_0_Relu_constraint_3[1,14]: Conv_0_Relu_ss[1,14]
   + 15.7113471031189 C630 <= 15.7113471031189
 Conv_0_Relu_constraint_3[1,15]: Conv_0_Relu_ss[1,15]
   + 4.252022132277489 C631 <= 4.252022132277489
 Conv_0_Relu_constraint_3[2,0]: Conv_0_Relu_ss[2,0]
   + 3.234174191951752 C632 <= 3.234174191951752
 Conv_0_Relu_constraint_3[2,1]: Conv_0_Relu_ss[2,1] + 11.7988965511322 C633
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[2,2]: Conv_0_Relu_ss[2,2]
   + 16.64243040978909 C634 <= 16.64243040978909
 Conv_0_Relu_constraint_3[2,3]: Conv_0_Relu_ss[2,3] + 14.8716185092926 C635
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[2,4]: Conv_0_Relu_ss[2,4]
   + 16.71291607618332 C636 <= 16.71291607618332
 Conv_0_Relu_constraint_3[2,5]: Conv_0_Relu_ss[2,5]
   + 15.65530395507812 C637 <= 15.65530395507812
 Conv_0_Relu_constraint_3[2,6]: Conv_0_Relu_ss[2,6]
   + 14.28846871852875 C638 <= 14.28846871852875
 Conv_0_Relu_constraint_3[2,7]: Conv_0_Relu_ss[2,7]
   + 16.01934917271137 C639 <= 16.01934917271137
 Conv_0_Relu_constraint_3[2,8]: Conv_0_Relu_ss[2,8] + 10.0236170142889 C640
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[2,9]: Conv_0_Relu_ss[2,9]
   + 12.89834886230528 C641 <= 12.89834886230528
 Conv_0_Relu_constraint_3[2,10]: Conv_0_Relu_ss[2,10]
   + 17.61239869892597 C642 <= 17.61239869892597
 Conv_0_Relu_constraint_3[2,11]: Conv_0_Relu_ss[2,11]
   + 17.77458453178406 C643 <= 17.77458453178406
 Conv_0_Relu_constraint_3[2,12]: Conv_0_Relu_ss[2,12]
   + 6.393570899963379 C644 <= 6.393570899963379
 Conv_0_Relu_constraint_3[2,13]: Conv_0_Relu_ss[2,13]
   + 8.590249866247177 C645 <= 8.590249866247177
 Conv_0_Relu_constraint_3[2,14]: Conv_0_Relu_ss[2,14]
   + 15.7113471031189 C646 <= 15.7113471031189
 Conv_0_Relu_constraint_3[2,15]: Conv_0_Relu_ss[2,15]
   + 4.252022132277489 C647 <= 4.252022132277489
 Conv_0_Relu_constraint_3[3,0]: Conv_0_Relu_ss[3,0]
   + 3.234174191951752 C648 <= 3.234174191951752
 Conv_0_Relu_constraint_3[3,1]: Conv_0_Relu_ss[3,1] + 11.7988965511322 C649
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[3,2]: Conv_0_Relu_ss[3,2]
   + 16.64243040978909 C650 <= 16.64243040978909
 Conv_0_Relu_constraint_3[3,3]: Conv_0_Relu_ss[3,3] + 14.8716185092926 C651
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[3,4]: Conv_0_Relu_ss[3,4]
   + 16.71291607618332 C652 <= 16.71291607618332
 Conv_0_Relu_constraint_3[3,5]: Conv_0_Relu_ss[3,5]
   + 15.65530395507812 C653 <= 15.65530395507812
 Conv_0_Relu_constraint_3[3,6]: Conv_0_Relu_ss[3,6]
   + 14.28846871852875 C654 <= 14.28846871852875
 Conv_0_Relu_constraint_3[3,7]: Conv_0_Relu_ss[3,7]
   + 16.01934917271137 C655 <= 16.01934917271137
 Conv_0_Relu_constraint_3[3,8]: Conv_0_Relu_ss[3,8] + 10.0236170142889 C656
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[3,9]: Conv_0_Relu_ss[3,9]
   + 12.89834886230528 C657 <= 12.89834886230528
 Conv_0_Relu_constraint_3[3,10]: Conv_0_Relu_ss[3,10]
   + 17.61239869892597 C658 <= 17.61239869892597
 Conv_0_Relu_constraint_3[3,11]: Conv_0_Relu_ss[3,11]
   + 17.77458453178406 C659 <= 17.77458453178406
 Conv_0_Relu_constraint_3[3,12]: Conv_0_Relu_ss[3,12]
   + 6.393570899963379 C660 <= 6.393570899963379
 Conv_0_Relu_constraint_3[3,13]: Conv_0_Relu_ss[3,13]
   + 8.590249866247177 C661 <= 8.590249866247177
 Conv_0_Relu_constraint_3[3,14]: Conv_0_Relu_ss[3,14]
   + 15.7113471031189 C662 <= 15.7113471031189
 Conv_0_Relu_constraint_3[3,15]: Conv_0_Relu_ss[3,15]
   + 4.252022132277489 C663 <= 4.252022132277489
 Conv_0_Relu_constraint_3[4,0]: Conv_0_Relu_ss[4,0]
   + 3.234174191951752 C664 <= 3.234174191951752
 Conv_0_Relu_constraint_3[4,1]: Conv_0_Relu_ss[4,1] + 11.7988965511322 C665
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[4,2]: Conv_0_Relu_ss[4,2]
   + 16.64243040978909 C666 <= 16.64243040978909
 Conv_0_Relu_constraint_3[4,3]: Conv_0_Relu_ss[4,3] + 14.8716185092926 C667
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[4,4]: Conv_0_Relu_ss[4,4]
   + 16.71291607618332 C668 <= 16.71291607618332
 Conv_0_Relu_constraint_3[4,5]: Conv_0_Relu_ss[4,5]
   + 15.65530395507812 C669 <= 15.65530395507812
 Conv_0_Relu_constraint_3[4,6]: Conv_0_Relu_ss[4,6]
   + 14.28846871852875 C670 <= 14.28846871852875
 Conv_0_Relu_constraint_3[4,7]: Conv_0_Relu_ss[4,7]
   + 16.01934917271137 C671 <= 16.01934917271137
 Conv_0_Relu_constraint_3[4,8]: Conv_0_Relu_ss[4,8] + 10.0236170142889 C672
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[4,9]: Conv_0_Relu_ss[4,9]
   + 12.89834886230528 C673 <= 12.89834886230528
 Conv_0_Relu_constraint_3[4,10]: Conv_0_Relu_ss[4,10]
   + 17.61239869892597 C674 <= 17.61239869892597
 Conv_0_Relu_constraint_3[4,11]: Conv_0_Relu_ss[4,11]
   + 17.77458453178406 C675 <= 17.77458453178406
 Conv_0_Relu_constraint_3[4,12]: Conv_0_Relu_ss[4,12]
   + 6.393570899963379 C676 <= 6.393570899963379
 Conv_0_Relu_constraint_3[4,13]: Conv_0_Relu_ss[4,13]
   + 8.590249866247177 C677 <= 8.590249866247177
 Conv_0_Relu_constraint_3[4,14]: Conv_0_Relu_ss[4,14]
   + 15.7113471031189 C678 <= 15.7113471031189
 Conv_0_Relu_constraint_3[4,15]: Conv_0_Relu_ss[4,15]
   + 4.252022132277489 C679 <= 4.252022132277489
 Conv_0_Relu_constraint_3[5,0]: Conv_0_Relu_ss[5,0]
   + 3.234174191951752 C680 <= 3.234174191951752
 Conv_0_Relu_constraint_3[5,1]: Conv_0_Relu_ss[5,1] + 11.7988965511322 C681
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[5,2]: Conv_0_Relu_ss[5,2]
   + 16.64243040978909 C682 <= 16.64243040978909
 Conv_0_Relu_constraint_3[5,3]: Conv_0_Relu_ss[5,3] + 14.8716185092926 C683
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[5,4]: Conv_0_Relu_ss[5,4]
   + 16.71291607618332 C684 <= 16.71291607618332
 Conv_0_Relu_constraint_3[5,5]: Conv_0_Relu_ss[5,5]
   + 15.65530395507812 C685 <= 15.65530395507812
 Conv_0_Relu_constraint_3[5,6]: Conv_0_Relu_ss[5,6]
   + 14.28846871852875 C686 <= 14.28846871852875
 Conv_0_Relu_constraint_3[5,7]: Conv_0_Relu_ss[5,7]
   + 16.01934917271137 C687 <= 16.01934917271137
 Conv_0_Relu_constraint_3[5,8]: Conv_0_Relu_ss[5,8] + 10.0236170142889 C688
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[5,9]: Conv_0_Relu_ss[5,9]
   + 12.89834886230528 C689 <= 12.89834886230528
 Conv_0_Relu_constraint_3[5,10]: Conv_0_Relu_ss[5,10]
   + 17.61239869892597 C690 <= 17.61239869892597
 Conv_0_Relu_constraint_3[5,11]: Conv_0_Relu_ss[5,11]
   + 17.77458453178406 C691 <= 17.77458453178406
 Conv_0_Relu_constraint_3[5,12]: Conv_0_Relu_ss[5,12]
   + 6.393570899963379 C692 <= 6.393570899963379
 Conv_0_Relu_constraint_3[5,13]: Conv_0_Relu_ss[5,13]
   + 8.590249866247177 C693 <= 8.590249866247177
 Conv_0_Relu_constraint_3[5,14]: Conv_0_Relu_ss[5,14]
   + 15.7113471031189 C694 <= 15.7113471031189
 Conv_0_Relu_constraint_3[5,15]: Conv_0_Relu_ss[5,15]
   + 4.252022132277489 C695 <= 4.252022132277489
 Conv_0_Relu_constraint_3[6,0]: Conv_0_Relu_ss[6,0]
   + 3.234174191951752 C696 <= 3.234174191951752
 Conv_0_Relu_constraint_3[6,1]: Conv_0_Relu_ss[6,1] + 11.7988965511322 C697
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[6,2]: Conv_0_Relu_ss[6,2]
   + 16.64243040978909 C698 <= 16.64243040978909
 Conv_0_Relu_constraint_3[6,3]: Conv_0_Relu_ss[6,3] + 14.8716185092926 C699
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[6,4]: Conv_0_Relu_ss[6,4]
   + 16.71291607618332 C700 <= 16.71291607618332
 Conv_0_Relu_constraint_3[6,5]: Conv_0_Relu_ss[6,5]
   + 15.65530395507812 C701 <= 15.65530395507812
 Conv_0_Relu_constraint_3[6,6]: Conv_0_Relu_ss[6,6]
   + 14.28846871852875 C702 <= 14.28846871852875
 Conv_0_Relu_constraint_3[6,7]: Conv_0_Relu_ss[6,7]
   + 16.01934917271137 C703 <= 16.01934917271137
 Conv_0_Relu_constraint_3[6,8]: Conv_0_Relu_ss[6,8] + 10.0236170142889 C704
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[6,9]: Conv_0_Relu_ss[6,9]
   + 12.89834886230528 C705 <= 12.89834886230528
 Conv_0_Relu_constraint_3[6,10]: Conv_0_Relu_ss[6,10]
   + 17.61239869892597 C706 <= 17.61239869892597
 Conv_0_Relu_constraint_3[6,11]: Conv_0_Relu_ss[6,11]
   + 17.77458453178406 C707 <= 17.77458453178406
 Conv_0_Relu_constraint_3[6,12]: Conv_0_Relu_ss[6,12]
   + 6.393570899963379 C708 <= 6.393570899963379
 Conv_0_Relu_constraint_3[6,13]: Conv_0_Relu_ss[6,13]
   + 8.590249866247177 C709 <= 8.590249866247177
 Conv_0_Relu_constraint_3[6,14]: Conv_0_Relu_ss[6,14]
   + 15.7113471031189 C710 <= 15.7113471031189
 Conv_0_Relu_constraint_3[6,15]: Conv_0_Relu_ss[6,15]
   + 4.252022132277489 C711 <= 4.252022132277489
 Conv_0_Relu_constraint_3[7,0]: Conv_0_Relu_ss[7,0]
   + 3.234174191951752 C712 <= 3.234174191951752
 Conv_0_Relu_constraint_3[7,1]: Conv_0_Relu_ss[7,1] + 11.7988965511322 C713
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[7,2]: Conv_0_Relu_ss[7,2]
   + 16.64243040978909 C714 <= 16.64243040978909
 Conv_0_Relu_constraint_3[7,3]: Conv_0_Relu_ss[7,3] + 14.8716185092926 C715
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[7,4]: Conv_0_Relu_ss[7,4]
   + 16.71291607618332 C716 <= 16.71291607618332
 Conv_0_Relu_constraint_3[7,5]: Conv_0_Relu_ss[7,5]
   + 15.65530395507812 C717 <= 15.65530395507812
 Conv_0_Relu_constraint_3[7,6]: Conv_0_Relu_ss[7,6]
   + 14.28846871852875 C718 <= 14.28846871852875
 Conv_0_Relu_constraint_3[7,7]: Conv_0_Relu_ss[7,7]
   + 16.01934917271137 C719 <= 16.01934917271137
 Conv_0_Relu_constraint_3[7,8]: Conv_0_Relu_ss[7,8] + 10.0236170142889 C720
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[7,9]: Conv_0_Relu_ss[7,9]
   + 12.89834886230528 C721 <= 12.89834886230528
 Conv_0_Relu_constraint_3[7,10]: Conv_0_Relu_ss[7,10]
   + 17.61239869892597 C722 <= 17.61239869892597
 Conv_0_Relu_constraint_3[7,11]: Conv_0_Relu_ss[7,11]
   + 17.77458453178406 C723 <= 17.77458453178406
 Conv_0_Relu_constraint_3[7,12]: Conv_0_Relu_ss[7,12]
   + 6.393570899963379 C724 <= 6.393570899963379
 Conv_0_Relu_constraint_3[7,13]: Conv_0_Relu_ss[7,13]
   + 8.590249866247177 C725 <= 8.590249866247177
 Conv_0_Relu_constraint_3[7,14]: Conv_0_Relu_ss[7,14]
   + 15.7113471031189 C726 <= 15.7113471031189
 Conv_0_Relu_constraint_3[7,15]: Conv_0_Relu_ss[7,15]
   + 4.252022132277489 C727 <= 4.252022132277489
 Conv_0_Relu_constraint_3[8,0]: Conv_0_Relu_ss[8,0]
   + 3.234174191951752 C728 <= 3.234174191951752
 Conv_0_Relu_constraint_3[8,1]: Conv_0_Relu_ss[8,1] + 11.7988965511322 C729
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[8,2]: Conv_0_Relu_ss[8,2]
   + 16.64243040978909 C730 <= 16.64243040978909
 Conv_0_Relu_constraint_3[8,3]: Conv_0_Relu_ss[8,3] + 14.8716185092926 C731
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[8,4]: Conv_0_Relu_ss[8,4]
   + 16.71291607618332 C732 <= 16.71291607618332
 Conv_0_Relu_constraint_3[8,5]: Conv_0_Relu_ss[8,5]
   + 15.65530395507812 C733 <= 15.65530395507812
 Conv_0_Relu_constraint_3[8,6]: Conv_0_Relu_ss[8,6]
   + 14.28846871852875 C734 <= 14.28846871852875
 Conv_0_Relu_constraint_3[8,7]: Conv_0_Relu_ss[8,7]
   + 16.01934917271137 C735 <= 16.01934917271137
 Conv_0_Relu_constraint_3[8,8]: Conv_0_Relu_ss[8,8] + 10.0236170142889 C736
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[8,9]: Conv_0_Relu_ss[8,9]
   + 12.89834886230528 C737 <= 12.89834886230528
 Conv_0_Relu_constraint_3[8,10]: Conv_0_Relu_ss[8,10]
   + 17.61239869892597 C738 <= 17.61239869892597
 Conv_0_Relu_constraint_3[8,11]: Conv_0_Relu_ss[8,11]
   + 17.77458453178406 C739 <= 17.77458453178406
 Conv_0_Relu_constraint_3[8,12]: Conv_0_Relu_ss[8,12]
   + 6.393570899963379 C740 <= 6.393570899963379
 Conv_0_Relu_constraint_3[8,13]: Conv_0_Relu_ss[8,13]
   + 8.590249866247177 C741 <= 8.590249866247177
 Conv_0_Relu_constraint_3[8,14]: Conv_0_Relu_ss[8,14]
   + 15.7113471031189 C742 <= 15.7113471031189
 Conv_0_Relu_constraint_3[8,15]: Conv_0_Relu_ss[8,15]
   + 4.252022132277489 C743 <= 4.252022132277489
 Conv_0_Relu_constraint_3[9,0]: Conv_0_Relu_ss[9,0]
   + 3.234174191951752 C744 <= 3.234174191951752
 Conv_0_Relu_constraint_3[9,1]: Conv_0_Relu_ss[9,1] + 11.7988965511322 C745
   <= 11.7988965511322
 Conv_0_Relu_constraint_3[9,2]: Conv_0_Relu_ss[9,2]
   + 16.64243040978909 C746 <= 16.64243040978909
 Conv_0_Relu_constraint_3[9,3]: Conv_0_Relu_ss[9,3] + 14.8716185092926 C747
   <= 14.8716185092926
 Conv_0_Relu_constraint_3[9,4]: Conv_0_Relu_ss[9,4]
   + 16.71291607618332 C748 <= 16.71291607618332
 Conv_0_Relu_constraint_3[9,5]: Conv_0_Relu_ss[9,5]
   + 15.65530395507812 C749 <= 15.65530395507812
 Conv_0_Relu_constraint_3[9,6]: Conv_0_Relu_ss[9,6]
   + 14.28846871852875 C750 <= 14.28846871852875
 Conv_0_Relu_constraint_3[9,7]: Conv_0_Relu_ss[9,7]
   + 16.01934917271137 C751 <= 16.01934917271137
 Conv_0_Relu_constraint_3[9,8]: Conv_0_Relu_ss[9,8] + 10.0236170142889 C752
   <= 10.0236170142889
 Conv_0_Relu_constraint_3[9,9]: Conv_0_Relu_ss[9,9]
   + 12.89834886230528 C753 <= 12.89834886230528
 Conv_0_Relu_constraint_3[9,10]: Conv_0_Relu_ss[9,10]
   + 17.61239869892597 C754 <= 17.61239869892597
 Conv_0_Relu_constraint_3[9,11]: Conv_0_Relu_ss[9,11]
   + 17.77458453178406 C755 <= 17.77458453178406
 Conv_0_Relu_constraint_3[9,12]: Conv_0_Relu_ss[9,12]
   + 6.393570899963379 C756 <= 6.393570899963379
 Conv_0_Relu_constraint_3[9,13]: Conv_0_Relu_ss[9,13]
   + 8.590249866247177 C757 <= 8.590249866247177
 Conv_0_Relu_constraint_3[9,14]: Conv_0_Relu_ss[9,14]
   + 15.7113471031189 C758 <= 15.7113471031189
 Conv_0_Relu_constraint_3[9,15]: Conv_0_Relu_ss[9,15]
   + 4.252022132277489 C759 <= 4.252022132277489
 Conv_1_output_constraint[0,0]: - 0.3061666786670685 Conv_0_Relu_ts[0,0]
   + 0.8854666948318481 Conv_0_Relu_ts[0,1]
   - 1.108363628387451 Conv_0_Relu_ts[0,2]
   - 0.0485447347164154 Conv_0_Relu_ts[0,3]
   - 0.3054571747779846 Conv_0_Relu_ts[0,4]
   - 1.666001558303833 Conv_0_Relu_ts[0,5]
   + 1.575538277626038 Conv_0_Relu_ts[0,6]
   - 0.8786032199859619 Conv_0_Relu_ts[0,7]
   + 0.1696695387363434 Conv_0_Relu_ts[0,8]
   - 0.2058112174272537 Conv_0_Relu_ts[0,9]
   - 0.8619717359542847 Conv_0_Relu_ts[0,10]
   + 1.268710613250732 Conv_0_Relu_ts[0,11]
   + 0.1005195453763008 Conv_0_Relu_ts[0,12]
   - 0.00627355137839913 Conv_0_Relu_ts[0,13]
   - 0.6727985143661499 Conv_0_Relu_ts[0,14]
   - 0.6082888841629028 Conv_0_Relu_ts[0,15]
   + 0.0746944323182106 Conv_1_aggregated_features[0,0]
   - 0.5317568182945251 Conv_1_aggregated_features[0,1]
   + 0.4405037760734558 Conv_1_aggregated_features[0,2]
   - 0.1488203704357147 Conv_1_aggregated_features[0,3]
   + 2.01956033706665 Conv_1_aggregated_features[0,4]
   + 0.8369019031524658 Conv_1_aggregated_features[0,5]
   + 0.1561367213726044 Conv_1_aggregated_features[0,6]
   + 0.1698544919490814 Conv_1_aggregated_features[0,7]
   - 0.0660476386547089 Conv_1_aggregated_features[0,8]
   + 1.570046067237854 Conv_1_aggregated_features[0,9]
   + 0.5540977120399475 Conv_1_aggregated_features[0,10]
   - 1.275410532951355 Conv_1_aggregated_features[0,11]
   - 0.092900924384594 Conv_1_aggregated_features[0,12]
   + 3.287018537521362 Conv_1_aggregated_features[0,13]
   + 2.392175912857056 Conv_1_aggregated_features[0,14]
   + 0.3933185040950775 Conv_1_aggregated_features[0,15] + Conv_1_t[0,0]
   = -2.192610263824463
 Conv_1_output_constraint[0,1]: 0.5202509164810181 Conv_0_Relu_ts[0,0]
   + 0.6686695218086243 Conv_0_Relu_ts[0,1]
   - 0.4720649719238281 Conv_0_Relu_ts[0,2]
   + 0.1026173532009125 Conv_0_Relu_ts[0,3]
   + 0.2593795359134674 Conv_0_Relu_ts[0,4]
   - 2.458898544311523 Conv_0_Relu_ts[0,5]
   - 0.5632196664810181 Conv_0_Relu_ts[0,6]
   - 1.791601657867432 Conv_0_Relu_ts[0,7]
   - 0.1268838942050934 Conv_0_Relu_ts[0,8]
   + 0.7120329737663269 Conv_0_Relu_ts[0,9]
   - 0.5212016701698303 Conv_0_Relu_ts[0,10]
   + 0.6136131286621094 Conv_0_Relu_ts[0,11]
   + 1.563590168952942 Conv_0_Relu_ts[0,12]
   - 0.3512922525405884 Conv_0_Relu_ts[0,13]
   + 0.5421575903892517 Conv_0_Relu_ts[0,14]
   + 0.7331249117851257 Conv_0_Relu_ts[0,15]
   + 0.0113700022920966 Conv_1_aggregated_features[0,0]
   + 2.169116258621216 Conv_1_aggregated_features[0,1]
   + 0.00623909616842866 Conv_1_aggregated_features[0,2]
   + 0.0937856733798981 Conv_1_aggregated_features[0,3]
   + 0.8328568339347839 Conv_1_aggregated_features[0,4]
   + 0.383074939250946 Conv_1_aggregated_features[0,5]
   - 0.1267539262771606 Conv_1_aggregated_features[0,6]
   + 0.6484018564224243 Conv_1_aggregated_features[0,7]
   + 0.1870926916599274 Conv_1_aggregated_features[0,8]
   - 0.4116691052913666 Conv_1_aggregated_features[0,9]
   + 0.1439376026391983 Conv_1_aggregated_features[0,10]
   + 1.05384635925293 Conv_1_aggregated_features[0,11]
   + 1.155439853668213 Conv_1_aggregated_features[0,12]
   - 0.2060132175683975 Conv_1_aggregated_features[0,13]
   + 0.6003169417381287 Conv_1_aggregated_features[0,14]
   + 0.1458597332239151 Conv_1_aggregated_features[0,15] + Conv_1_t[0,1]
   = -0.1077260375022888
 Conv_1_output_constraint[0,2]: 0.4269110858440399 Conv_0_Relu_ts[0,0]
   - 1.097789287567139 Conv_0_Relu_ts[0,1]
   - 0.3380239307880402 Conv_0_Relu_ts[0,2]
   + 0.23756343126297 Conv_0_Relu_ts[0,3]
   - 0.337738424539566 Conv_0_Relu_ts[0,4]
   - 1.407688736915588 Conv_0_Relu_ts[0,5]
   - 0.7854894995689392 Conv_0_Relu_ts[0,6]
   - 1.080899715423584 Conv_0_Relu_ts[0,7]
   + 0.0233876463025808 Conv_0_Relu_ts[0,8]
   + 1.849825739860535 Conv_0_Relu_ts[0,9]
   + 0.4580322206020355 Conv_0_Relu_ts[0,10]
   - 0.819361686706543 Conv_0_Relu_ts[0,11]
   - 0.0822571739554405 Conv_0_Relu_ts[0,12]
   - 0.1501271426677704 Conv_0_Relu_ts[0,13]
   + 0.1831801384687424 Conv_0_Relu_ts[0,14]
   + 0.00308510870672762 Conv_0_Relu_ts[0,15]
   - 0.2153502702713013 Conv_1_aggregated_features[0,0]
   + 3.817557334899902 Conv_1_aggregated_features[0,1]
   - 0.3402683734893799 Conv_1_aggregated_features[0,2]
   + 0.0650523602962494 Conv_1_aggregated_features[0,3]
   - 0.4435569643974304 Conv_1_aggregated_features[0,4]
   + 0.045248981565237 Conv_1_aggregated_features[0,5]
   + 0.844337522983551 Conv_1_aggregated_features[0,6]
   - 0.0691490024328232 Conv_1_aggregated_features[0,7]
   + 0.0830505788326263 Conv_1_aggregated_features[0,8]
   - 0.3022302687168121 Conv_1_aggregated_features[0,9]
   - 0.611566960811615 Conv_1_aggregated_features[0,10]
   + 3.582128524780273 Conv_1_aggregated_features[0,11]
   + 0.1314867585897446 Conv_1_aggregated_features[0,12]
   + 0.5333523154258728 Conv_1_aggregated_features[0,13]
   - 0.1983102858066559 Conv_1_aggregated_features[0,14]
   + 0.1628565639257431 Conv_1_aggregated_features[0,15] + Conv_1_t[0,2]
   = -0.4443503022193909
 Conv_1_output_constraint[0,3]: 0.066770151257515 Conv_0_Relu_ts[0,0]
   + 0.3136637210845947 Conv_0_Relu_ts[0,1]
   + 0.2078424692153931 Conv_0_Relu_ts[0,2]
   - 0.1673517823219299 Conv_0_Relu_ts[0,3]
   + 1.022224068641663 Conv_0_Relu_ts[0,4]
   - 0.1723377108573914 Conv_0_Relu_ts[0,5]
   + 0.3032898306846619 Conv_0_Relu_ts[0,6]
   + 0.0841449126601219 Conv_0_Relu_ts[0,7]
   + 0.0921086072921753 Conv_0_Relu_ts[0,8]
   - 0.022469462826848 Conv_0_Relu_ts[0,9]
   - 0.1657154262065887 Conv_0_Relu_ts[0,10]
   + 1.007031679153442 Conv_0_Relu_ts[0,11]
   + 0.0208180602639914 Conv_0_Relu_ts[0,12]
   + 0.3933880031108856 Conv_0_Relu_ts[0,13]
   + 0.3077191412448883 Conv_0_Relu_ts[0,14]
   + 0.0400979965925217 Conv_0_Relu_ts[0,15]
   + 0.0230858232825994 Conv_1_aggregated_features[0,0]
   + 0.3792195022106171 Conv_1_aggregated_features[0,1]
   + 0.1572305560112 Conv_1_aggregated_features[0,2]
   - 0.0165807008743286 Conv_1_aggregated_features[0,3]
   + 0.5649795532226562 Conv_1_aggregated_features[0,4]
   + 0.6253551840782166 Conv_1_aggregated_features[0,5]
   + 0.3968201577663422 Conv_1_aggregated_features[0,6]
   + 0.2728661298751831 Conv_1_aggregated_features[0,7]
   - 0.210608571767807 Conv_1_aggregated_features[0,8]
   + 0.094737209379673 Conv_1_aggregated_features[0,9]
   - 0.1354089975357056 Conv_1_aggregated_features[0,10]
   + 0.9330839514732361 Conv_1_aggregated_features[0,11]
   + 0.1677136421203613 Conv_1_aggregated_features[0,12]
   + 0.6573846936225891 Conv_1_aggregated_features[0,13]
   + 0.044663343578577 Conv_1_aggregated_features[0,14]
   - 0.0132347755134106 Conv_1_aggregated_features[0,15] + Conv_1_t[0,3]
   = -0.327608048915863
 Conv_1_output_constraint[0,4]: - 0.0720538049936295 Conv_0_Relu_ts[0,0]
   + 3.730369091033936 Conv_0_Relu_ts[0,1]
   - 0.8118388652801514 Conv_0_Relu_ts[0,2]
   - 0.174347460269928 Conv_0_Relu_ts[0,3]
   - 3.084089279174805 Conv_0_Relu_ts[0,4]
   - 5.146248340606689 Conv_0_Relu_ts[0,5]
   - 2.440467357635498 Conv_0_Relu_ts[0,6]
   - 3.765322923660278 Conv_0_Relu_ts[0,7]
   + 0.153628796339035 Conv_0_Relu_ts[0,8]
   + 0.1013353765010834 Conv_0_Relu_ts[0,9]
   - 0.5803871154785156 Conv_0_Relu_ts[0,10]
   - 2.657181739807129 Conv_0_Relu_ts[0,11]
   + 0.6840037703514099 Conv_0_Relu_ts[0,12]
   - 1.184461355209351 Conv_0_Relu_ts[0,13]
   - 0.3772841989994049 Conv_0_Relu_ts[0,14]
   + 0.5856359601020813 Conv_0_Relu_ts[0,15]
   + 0.2062403708696365 Conv_1_aggregated_features[0,0]
   + 1.470675468444824 Conv_1_aggregated_features[0,1]
   + 0.3523006737232208 Conv_1_aggregated_features[0,2]
   - 0.1351108849048615 Conv_1_aggregated_features[0,3]
   + 0.8932182788848877 Conv_1_aggregated_features[0,4]
   + 1.194340229034424 Conv_1_aggregated_features[0,5]
   + 0.0163087919354439 Conv_1_aggregated_features[0,6]
   - 0.5082771182060242 Conv_1_aggregated_features[0,7]
   - 0.0201514065265656 Conv_1_aggregated_features[0,8]
   + 0.9045064449310303 Conv_1_aggregated_features[0,9]
   + 0.7243757843971252 Conv_1_aggregated_features[0,10]
   + 0.5823414921760559 Conv_1_aggregated_features[0,11]
   + 0.0738087296485901 Conv_1_aggregated_features[0,12]
   - 0.3080458641052246 Conv_1_aggregated_features[0,13]
   + 0.9457569122314453 Conv_1_aggregated_features[0,14]
   + 0.4423830211162567 Conv_1_aggregated_features[0,15] + Conv_1_t[0,4]
   = -0.2138544023036957
 Conv_1_output_constraint[0,5]: 0.2116706073284149 Conv_0_Relu_ts[0,0]
   + 0.2319677472114563 Conv_0_Relu_ts[0,1]
   + 0.1967353820800781 Conv_0_Relu_ts[0,2]
   - 0.1974936425685883 Conv_0_Relu_ts[0,3]
   + 0.087493434548378 Conv_0_Relu_ts[0,4]
   + 0.0226023308932781 Conv_0_Relu_ts[0,5]
   - 0.1455883532762527 Conv_0_Relu_ts[0,6]
   - 0.1098284423351288 Conv_0_Relu_ts[0,7]
   + 0.0375414304435253 Conv_0_Relu_ts[0,8]
   + 0.2209681868553162 Conv_0_Relu_ts[0,9]
   + 0.0180207490921021 Conv_0_Relu_ts[0,10]
   - 0.0944192707538605 Conv_0_Relu_ts[0,11]
   + 0.1318628787994385 Conv_0_Relu_ts[0,12]
   + 0.1718421876430511 Conv_0_Relu_ts[0,13]
   - 0.0992768108844757 Conv_0_Relu_ts[0,14]
   + 0.2220577001571655 Conv_0_Relu_ts[0,15]
   + 0.0500434041023254 Conv_1_aggregated_features[0,0]
   + 0.2417693138122559 Conv_1_aggregated_features[0,1]
   + 0.1884889900684357 Conv_1_aggregated_features[0,2]
   - 0.0459566116333008 Conv_1_aggregated_features[0,3]
   - 0.1095028519630432 Conv_1_aggregated_features[0,4]
   - 0.1494371891021729 Conv_1_aggregated_features[0,5]
   - 0.0601528584957123 Conv_1_aggregated_features[0,6]
   - 0.2431226968765259 Conv_1_aggregated_features[0,7]
   + 0.0709281861782074 Conv_1_aggregated_features[0,8]
   + 0.1859873235225677 Conv_1_aggregated_features[0,9]
   - 0.0025022029876709 Conv_1_aggregated_features[0,10]
   + 0.1165908575057983 Conv_1_aggregated_features[0,11]
   - 0.0946725904941559 Conv_1_aggregated_features[0,12]
   + 0.1679102778434753 Conv_1_aggregated_features[0,13]
   + 0.0888539850711823 Conv_1_aggregated_features[0,14]
   + 0.0481749773025513 Conv_1_aggregated_features[0,15] + Conv_1_t[0,5]
   = -0.2017861604690552
 Conv_1_output_constraint[0,6]: - 0.5161762237548828 Conv_0_Relu_ts[0,0]
   - 0.9039249420166016 Conv_0_Relu_ts[0,1]
   - 0.5091935992240906 Conv_0_Relu_ts[0,2]
   + 0.177293062210083 Conv_0_Relu_ts[0,3]
   + 1.534894466400146 Conv_0_Relu_ts[0,4]
   + 0.3099820613861084 Conv_0_Relu_ts[0,5]
   + 1.335353136062622 Conv_0_Relu_ts[0,6]
   - 0.3965733051300049 Conv_0_Relu_ts[0,7]
   + 0.0226722583174706 Conv_0_Relu_ts[0,8]
   - 0.0153551828116179 Conv_0_Relu_ts[0,9]
   - 0.188335582613945 Conv_0_Relu_ts[0,10]
   + 0.1465377509593964 Conv_0_Relu_ts[0,11]
   - 0.4378408491611481 Conv_0_Relu_ts[0,12]
   + 0.5950998067855835 Conv_0_Relu_ts[0,13]
   - 0.2773410975933075 Conv_0_Relu_ts[0,14]
   - 0.3855878710746765 Conv_0_Relu_ts[0,15]
   + 0.0145458607003093 Conv_1_aggregated_features[0,0]
   - 1.349490284919739 Conv_1_aggregated_features[0,1]
   - 0.6738612055778503 Conv_1_aggregated_features[0,2]
   - 0.1977581083774567 Conv_1_aggregated_features[0,3]
   + 3.616869688034058 Conv_1_aggregated_features[0,4]
   - 0.5846084952354431 Conv_1_aggregated_features[0,5]
   + 0.3911246955394745 Conv_1_aggregated_features[0,6]
   - 0.2603789269924164 Conv_1_aggregated_features[0,7]
   + 0.0530429184436798 Conv_1_aggregated_features[0,8]
   + 0.7039647698402405 Conv_1_aggregated_features[0,9]
   - 0.4948081970214844 Conv_1_aggregated_features[0,10]
   - 1.33747124671936 Conv_1_aggregated_features[0,11]
   + 0.4521820247173309 Conv_1_aggregated_features[0,12]
   + 2.879487276077271 Conv_1_aggregated_features[0,13]
   + 0.6494181156158447 Conv_1_aggregated_features[0,14]
   - 0.0441654324531555 Conv_1_aggregated_features[0,15] + Conv_1_t[0,6]
   = -1.141431212425232
 Conv_1_output_constraint[0,7]: 0.1232976838946342 Conv_0_Relu_ts[0,0]
   + 0.7059634327888489 Conv_0_Relu_ts[0,1]
   + 0.0986589044332504 Conv_0_Relu_ts[0,2]
   - 0.1159188747406006 Conv_0_Relu_ts[0,3]
   + 0.1207201555371284 Conv_0_Relu_ts[0,4]
   + 0.4950260519981384 Conv_0_Relu_ts[0,5]
   - 0.1985135078430176 Conv_0_Relu_ts[0,6]
   + 0.3846564292907715 Conv_0_Relu_ts[0,7]
   - 0.2091868817806244 Conv_0_Relu_ts[0,8]
   + 0.0282646510750055 Conv_0_Relu_ts[0,9]
   + 0.00616153934970498 Conv_0_Relu_ts[0,10]
   - 0.2992357313632965 Conv_0_Relu_ts[0,11]
   + 0.6717308759689331 Conv_0_Relu_ts[0,12]
   + 0.3884595632553101 Conv_0_Relu_ts[0,13]
   + 0.1088093742728233 Conv_0_Relu_ts[0,14]
   + 0.1363743990659714 Conv_0_Relu_ts[0,15]
   + 0.2181874513626099 Conv_1_aggregated_features[0,0]
   + 0.1884859055280685 Conv_1_aggregated_features[0,1]
   + 0.1986039727926254 Conv_1_aggregated_features[0,2]
   - 0.0569350719451904 Conv_1_aggregated_features[0,3]
   + 0.4957895576953888 Conv_1_aggregated_features[0,4]
   - 0.2215360552072525 Conv_1_aggregated_features[0,5]
   - 0.3126825392246246 Conv_1_aggregated_features[0,6]
   + 0.2129529714584351 Conv_1_aggregated_features[0,7]
   + 0.1905750632286072 Conv_1_aggregated_features[0,8]
   + 0.3835417926311493 Conv_1_aggregated_features[0,9]
   + 0.0515264272689819 Conv_1_aggregated_features[0,10]
   + 0.2774861454963684 Conv_1_aggregated_features[0,11]
   + 0.1720549166202545 Conv_1_aggregated_features[0,12]
   + 0.2042496353387833 Conv_1_aggregated_features[0,13]
   + 0.1717791706323624 Conv_1_aggregated_features[0,14]
   - 0.0635386854410172 Conv_1_aggregated_features[0,15] + Conv_1_t[0,7]
   = -0.1961010098457336
 Conv_1_output_constraint[0,8]: - 0.0984712317585945 Conv_0_Relu_ts[0,0]
   - 0.8972246050834656 Conv_0_Relu_ts[0,1]
   - 0.1342054009437561 Conv_0_Relu_ts[0,2]
   + 0.0934775173664093 Conv_0_Relu_ts[0,3]
   + 3.929172039031982 Conv_0_Relu_ts[0,4]
   - 1.699812531471252 Conv_0_Relu_ts[0,5]
   + 0.4780299663543701 Conv_0_Relu_ts[0,6]
   + 0.1916014850139618 Conv_0_Relu_ts[0,7]
   - 0.0282735247164965 Conv_0_Relu_ts[0,8]
   - 0.10334312915802 Conv_0_Relu_ts[0,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[0,10]
   - 0.9000989198684692 Conv_0_Relu_ts[0,11]
   - 0.3381493389606476 Conv_0_Relu_ts[0,12]
   + 0.2327869236469269 Conv_0_Relu_ts[0,13]
   + 0.6162272691726685 Conv_0_Relu_ts[0,14]
   - 0.1045716181397438 Conv_0_Relu_ts[0,15]
   - 0.0895406752824783 Conv_1_aggregated_features[0,0]
   - 0.0733596011996269 Conv_1_aggregated_features[0,1]
   - 0.6064443588256836 Conv_1_aggregated_features[0,2]
   + 0.1685465276241302 Conv_1_aggregated_features[0,3]
   + 2.521227836608887 Conv_1_aggregated_features[0,4]
   + 0.8868938088417053 Conv_1_aggregated_features[0,5]
   - 0.3730944097042084 Conv_1_aggregated_features[0,6]
   + 0.4433300495147705 Conv_1_aggregated_features[0,7]
   + 0.2465403079986572 Conv_1_aggregated_features[0,8]
   + 0.1199371814727783 Conv_1_aggregated_features[0,9]
   - 0.427333652973175 Conv_1_aggregated_features[0,10]
   - 1.286244511604309 Conv_1_aggregated_features[0,11]
   + 0.4271916449069977 Conv_1_aggregated_features[0,12]
   + 1.475327014923096 Conv_1_aggregated_features[0,13]
   - 0.2351342439651489 Conv_1_aggregated_features[0,14]
   - 0.1273849010467529 Conv_1_aggregated_features[0,15] + Conv_1_t[0,8]
   = -1.303357601165771
 Conv_1_output_constraint[0,9]: 0.1534582376480103 Conv_0_Relu_ts[0,0]
   + 1.456407189369202 Conv_0_Relu_ts[0,1]
   - 0.0228729248046875 Conv_0_Relu_ts[0,2]
   + 0.1227743625640869 Conv_0_Relu_ts[0,3]
   + 0.7793847918510437 Conv_0_Relu_ts[0,4]
   - 0.2757052779197693 Conv_0_Relu_ts[0,5]
   + 0.6606881022453308 Conv_0_Relu_ts[0,6]
   - 1.107266664505005 Conv_0_Relu_ts[0,7]
   + 0.1723826229572296 Conv_0_Relu_ts[0,8]
   - 0.5540673732757568 Conv_0_Relu_ts[0,9]
   + 0.3133971095085144 Conv_0_Relu_ts[0,10]
   + 0.9318023324012756 Conv_0_Relu_ts[0,11]
   + 0.5855908989906311 Conv_0_Relu_ts[0,12]
   - 0.2116718590259552 Conv_0_Relu_ts[0,13]
   - 0.3693177700042725 Conv_0_Relu_ts[0,14]
   - 0.0642165616154671 Conv_0_Relu_ts[0,15]
   + 0.1819062232971191 Conv_1_aggregated_features[0,0]
   + 1.311912417411804 Conv_1_aggregated_features[0,1]
   + 0.3460695445537567 Conv_1_aggregated_features[0,2]
   + 0.1086616516113281 Conv_1_aggregated_features[0,3]
   - 0.6099227070808411 Conv_1_aggregated_features[0,4]
   + 0.3786990940570831 Conv_1_aggregated_features[0,5]
   + 0.9473716020584106 Conv_1_aggregated_features[0,6]
   + 0.2196423262357712 Conv_1_aggregated_features[0,7]
   + 0.0830722749233246 Conv_1_aggregated_features[0,8]
   - 0.4822347462177277 Conv_1_aggregated_features[0,9]
   + 0.5108910202980042 Conv_1_aggregated_features[0,10]
   + 3.214276790618896 Conv_1_aggregated_features[0,11]
   - 0.7224436402320862 Conv_1_aggregated_features[0,12]
   + 0.2678262591362 Conv_1_aggregated_features[0,13]
   - 0.8807565569877625 Conv_1_aggregated_features[0,14]
   - 0.3268090188503265 Conv_1_aggregated_features[0,15] + Conv_1_t[0,9]
   = -0.8283404111862183
 Conv_1_output_constraint[0,10]: 0.3653882741928101 Conv_0_Relu_ts[0,0]
   - 0.8208209872245789 Conv_0_Relu_ts[0,1]
   + 0.3088222146034241 Conv_0_Relu_ts[0,2]
   + 0.1173163652420044 Conv_0_Relu_ts[0,3]
   + 0.658578634262085 Conv_0_Relu_ts[0,4]
   + 0.584019660949707 Conv_0_Relu_ts[0,5]
   - 0.8654046058654785 Conv_0_Relu_ts[0,6]
   + 0.3697350919246674 Conv_0_Relu_ts[0,7]
   - 0.2274059355258942 Conv_0_Relu_ts[0,8]
   + 1.197938442230225 Conv_0_Relu_ts[0,9]
   + 0.2446341812610626 Conv_0_Relu_ts[0,10]
   - 0.5676743984222412 Conv_0_Relu_ts[0,11]
   - 0.3123646676540375 Conv_0_Relu_ts[0,12]
   + 3.181503534317017 Conv_0_Relu_ts[0,13]
   + 0.6926240921020508 Conv_0_Relu_ts[0,14]
   + 0.0886927545070648 Conv_0_Relu_ts[0,15]
   - 0.2005636394023895 Conv_1_aggregated_features[0,0]
   + 1.403087973594666 Conv_1_aggregated_features[0,1]
   + 0.0523448958992958 Conv_1_aggregated_features[0,2]
   + 0.2411543428897858 Conv_1_aggregated_features[0,3]
   + 0.5953171253204346 Conv_1_aggregated_features[0,4]
   - 0.8278964757919312 Conv_1_aggregated_features[0,5]
   + 0.2499217540025711 Conv_1_aggregated_features[0,6]
   - 0.7393644452095032 Conv_1_aggregated_features[0,7]
   + 0.0902875959873199 Conv_1_aggregated_features[0,8]
   + 0.1488711386919022 Conv_1_aggregated_features[0,9]
   - 0.0303693022578955 Conv_1_aggregated_features[0,10]
   + 2.024244070053101 Conv_1_aggregated_features[0,11]
   + 0.495431512594223 Conv_1_aggregated_features[0,12]
   + 0.3463623225688934 Conv_1_aggregated_features[0,13]
   - 0.0763808861374855 Conv_1_aggregated_features[0,14]
   + 0.1597562879323959 Conv_1_aggregated_features[0,15] + Conv_1_t[0,10]
   = -0.5063248872756958
 Conv_1_output_constraint[0,11]: - 0.5185666680335999 Conv_0_Relu_ts[0,0]
   + 3.656918525695801 Conv_0_Relu_ts[0,1]
   - 0.7082476615905762 Conv_0_Relu_ts[0,2]
   + 0.1922211349010468 Conv_0_Relu_ts[0,3]
   - 0.7149283289909363 Conv_0_Relu_ts[0,4]
   - 3.255658626556396 Conv_0_Relu_ts[0,5]
   + 2.464392185211182 Conv_0_Relu_ts[0,6]
   - 0.9477378129959106 Conv_0_Relu_ts[0,7]
   + 0.168237030506134 Conv_0_Relu_ts[0,8]
   - 1.177640557289124 Conv_0_Relu_ts[0,9]
   - 0.7062549591064453 Conv_0_Relu_ts[0,10]
   + 1.453806400299072 Conv_0_Relu_ts[0,11]
   + 0.2612711191177368 Conv_0_Relu_ts[0,12]
   + 0.2999382615089417 Conv_0_Relu_ts[0,13]
   - 1.255579590797424 Conv_0_Relu_ts[0,14]
   - 0.4265820384025574 Conv_0_Relu_ts[0,15]
   + 0.3379200994968414 Conv_1_aggregated_features[0,0]
   - 1.33229672908783 Conv_1_aggregated_features[0,1]
   - 0.1594242304563522 Conv_1_aggregated_features[0,2]
   + 0.2298935651779175 Conv_1_aggregated_features[0,3]
   - 1.311494588851929 Conv_1_aggregated_features[0,4]
   + 3.605161666870117 Conv_1_aggregated_features[0,5]
   - 0.3176752626895905 Conv_1_aggregated_features[0,6]
   + 0.3926282525062561 Conv_1_aggregated_features[0,7]
   + 0.1479305028915405 Conv_1_aggregated_features[0,8]
   - 0.0372937023639679 Conv_1_aggregated_features[0,9]
   + 0.3455427587032318 Conv_1_aggregated_features[0,10]
   - 2.002299070358276 Conv_1_aggregated_features[0,11]
   - 0.9904713034629822 Conv_1_aggregated_features[0,12]
   + 0.0751789212226868 Conv_1_aggregated_features[0,13]
   - 0.5166233777999878 Conv_1_aggregated_features[0,14]
   + 0.0510688796639442 Conv_1_aggregated_features[0,15] + Conv_1_t[0,11]
   = -1.480785727500916
 Conv_1_output_constraint[0,12]: 0.3080185949802399 Conv_0_Relu_ts[0,0]
   + 0.0994751453399658 Conv_0_Relu_ts[0,1]
   + 0.4488444924354553 Conv_0_Relu_ts[0,2]
   + 0.0305773317813873 Conv_0_Relu_ts[0,3]
   - 1.673393130302429 Conv_0_Relu_ts[0,4]
   + 0.2824500501155853 Conv_0_Relu_ts[0,5]
   - 1.192709684371948 Conv_0_Relu_ts[0,6]
   + 1.454243063926697 Conv_0_Relu_ts[0,7]
   + 0.1514801830053329 Conv_0_Relu_ts[0,8]
   + 1.026351928710938 Conv_0_Relu_ts[0,9]
   + 0.1183762699365616 Conv_0_Relu_ts[0,10]
   + 0.352224588394165 Conv_0_Relu_ts[0,11]
   + 1.090276837348938 Conv_0_Relu_ts[0,12]
   - 2.053535461425781 Conv_0_Relu_ts[0,13]
   + 2.538641452789307 Conv_0_Relu_ts[0,14]
   + 0.6387560963630676 Conv_0_Relu_ts[0,15]
   - 0.1990496516227722 Conv_1_aggregated_features[0,0]
   + 1.247365713119507 Conv_1_aggregated_features[0,1]
   + 0.2213435918092728 Conv_1_aggregated_features[0,2]
   + 0.2398849725723267 Conv_1_aggregated_features[0,3]
   - 2.001479864120483 Conv_1_aggregated_features[0,4]
   - 1.084800243377686 Conv_1_aggregated_features[0,5]
   + 0.4287458956241608 Conv_1_aggregated_features[0,6]
   - 0.8179780840873718 Conv_1_aggregated_features[0,7]
   - 0.0883614718914032 Conv_1_aggregated_features[0,8]
   - 0.4398066997528076 Conv_1_aggregated_features[0,9]
   + 0.2760448455810547 Conv_1_aggregated_features[0,10]
   + 1.944519639015198 Conv_1_aggregated_features[0,11]
   + 1.041437149047852 Conv_1_aggregated_features[0,12]
   - 1.682050704956055 Conv_1_aggregated_features[0,13]
   - 0.1825132220983505 Conv_1_aggregated_features[0,14]
   + 0.2881160080432892 Conv_1_aggregated_features[0,15] + Conv_1_t[0,12]
   = 2.457676649093628
 Conv_1_output_constraint[0,13]: 0.2853958606719971 Conv_0_Relu_ts[0,0]
   + 0.0445350483059883 Conv_0_Relu_ts[0,1]
   - 0.0597315393388271 Conv_0_Relu_ts[0,2]
   + 0.2403226792812347 Conv_0_Relu_ts[0,3]
   + 0.5103238224983215 Conv_0_Relu_ts[0,4]
   - 0.1369927227497101 Conv_0_Relu_ts[0,5]
   + 0.5091087818145752 Conv_0_Relu_ts[0,6]
   - 0.0576926171779633 Conv_0_Relu_ts[0,7]
   - 0.00701320171356201 Conv_0_Relu_ts[0,8]
   + 0.5782948136329651 Conv_0_Relu_ts[0,9]
   - 0.1188536584377289 Conv_0_Relu_ts[0,10]
   - 0.078026682138443 Conv_0_Relu_ts[0,11]
   + 0.1965750753879547 Conv_0_Relu_ts[0,12]
   + 0.4982857704162598 Conv_0_Relu_ts[0,13]
   - 0.0995257869362831 Conv_0_Relu_ts[0,14]
   + 0.0712818428874016 Conv_0_Relu_ts[0,15]
   + 0.2628078460693359 Conv_1_aggregated_features[0,0]
   + 0.6577805280685425 Conv_1_aggregated_features[0,1]
   - 0.1415485441684723 Conv_1_aggregated_features[0,2]
   + 0.1597791016101837 Conv_1_aggregated_features[0,3]
   + 0.3969745337963104 Conv_1_aggregated_features[0,4]
   - 0.3941676616668701 Conv_1_aggregated_features[0,5]
   + 0.3484204709529877 Conv_1_aggregated_features[0,6]
   + 0.2601790428161621 Conv_1_aggregated_features[0,7]
   - 0.0652478933334351 Conv_1_aggregated_features[0,8]
   + 0.3492280542850494 Conv_1_aggregated_features[0,9]
   + 0.1917867064476013 Conv_1_aggregated_features[0,10]
   + 0.2439181953668594 Conv_1_aggregated_features[0,11]
   + 0.1009163707494736 Conv_1_aggregated_features[0,12]
   + 0.2350947707891464 Conv_1_aggregated_features[0,13]
   + 0.2627962231636047 Conv_1_aggregated_features[0,14]
   - 0.0959920063614845 Conv_1_aggregated_features[0,15] + Conv_1_t[0,13]
   = -0.4268372058868408
 Conv_1_output_constraint[0,14]: - 0.2448655962944031 Conv_0_Relu_ts[0,0]
   + 0.2688764035701752 Conv_0_Relu_ts[0,1]
   + 0.0118266940116882 Conv_0_Relu_ts[0,2]
   - 0.0670525431632996 Conv_0_Relu_ts[0,3]
   + 0.1511579602956772 Conv_0_Relu_ts[0,4]
   - 0.4796198606491089 Conv_0_Relu_ts[0,5]
   - 0.1699391901493073 Conv_0_Relu_ts[0,6]
   + 0.2647148370742798 Conv_0_Relu_ts[0,7]
   + 0.137117475271225 Conv_0_Relu_ts[0,8]
   + 0.1737731248140335 Conv_0_Relu_ts[0,9]
   - 0.1610838174819946 Conv_0_Relu_ts[0,10]
   - 0.6311220526695251 Conv_0_Relu_ts[0,11]
   + 0.6263757348060608 Conv_0_Relu_ts[0,12]
   - 0.4823384284973145 Conv_0_Relu_ts[0,13]
   + 0.2192765176296234 Conv_0_Relu_ts[0,14]
   + 0.1739131808280945 Conv_0_Relu_ts[0,15]
   + 0.0443203188478947 Conv_1_aggregated_features[0,0]
   + 0.7668815851211548 Conv_1_aggregated_features[0,1]
   + 0.248365581035614 Conv_1_aggregated_features[0,2]
   - 0.2433404624462128 Conv_1_aggregated_features[0,3]
   + 0.2770798802375793 Conv_1_aggregated_features[0,4]
   - 1.224791765213013 Conv_1_aggregated_features[0,5]
   - 0.1892879456281662 Conv_1_aggregated_features[0,6]
   + 0.1141589805483818 Conv_1_aggregated_features[0,7]
   + 0.2022489905357361 Conv_1_aggregated_features[0,8]
   - 0.0391396470367908 Conv_1_aggregated_features[0,9]
   - 0.0935317650437355 Conv_1_aggregated_features[0,10]
   - 0.5407891273498535 Conv_1_aggregated_features[0,11]
   + 0.3130630552768707 Conv_1_aggregated_features[0,12]
   + 0.3410882651805878 Conv_1_aggregated_features[0,13]
   - 1.855417490005493 Conv_1_aggregated_features[0,14]
   + 0.247613787651062 Conv_1_aggregated_features[0,15] + Conv_1_t[0,14]
   = -0.2306040376424789
 Conv_1_output_constraint[0,15]: 0.043729841709137 Conv_0_Relu_ts[0,0]
   + 0.045797374099493 Conv_0_Relu_ts[0,1]
   - 0.1762047111988068 Conv_0_Relu_ts[0,2]
   - 0.2225021123886108 Conv_0_Relu_ts[0,3]
   + 0.1344186365604401 Conv_0_Relu_ts[0,4]
   + 0.2975046932697296 Conv_0_Relu_ts[0,5]
   + 0.1135610193014145 Conv_0_Relu_ts[0,6]
   + 0.165085032582283 Conv_0_Relu_ts[0,7]
   + 0.0632660016417503 Conv_0_Relu_ts[0,8]
   + 0.2534148395061493 Conv_0_Relu_ts[0,9]
   + 0.1100017726421356 Conv_0_Relu_ts[0,10]
   + 0.045901358127594 Conv_0_Relu_ts[0,11]
   + 0.2102706432342529 Conv_0_Relu_ts[0,12]
   + 0.1767551600933075 Conv_0_Relu_ts[0,13]
   - 0.0864512398838997 Conv_0_Relu_ts[0,14]
   + 0.1616824418306351 Conv_0_Relu_ts[0,15]
   + 0.0113421082496643 Conv_1_aggregated_features[0,0]
   + 0.2387615889310837 Conv_1_aggregated_features[0,1]
   + 0.1763307154178619 Conv_1_aggregated_features[0,2]
   + 0.1442219316959381 Conv_1_aggregated_features[0,3]
   + 0.3573774099349976 Conv_1_aggregated_features[0,4]
   - 0.0406427308917046 Conv_1_aggregated_features[0,5]
   + 0.1155626475811005 Conv_1_aggregated_features[0,6]
   + 0.0186243671923876 Conv_1_aggregated_features[0,7]
   - 0.1389756500720978 Conv_1_aggregated_features[0,8]
   + 0.274962306022644 Conv_1_aggregated_features[0,9]
   + 0.0309249386191368 Conv_1_aggregated_features[0,10]
   - 0.1606657803058624 Conv_1_aggregated_features[0,11]
   + 0.2456246316432953 Conv_1_aggregated_features[0,12]
   + 0.4163841009140015 Conv_1_aggregated_features[0,13]
   + 0.29454505443573 Conv_1_aggregated_features[0,14]
   + 0.1688967496156693 Conv_1_aggregated_features[0,15] + Conv_1_t[0,15]
   = -0.112743653357029
 Conv_1_output_constraint[1,0]: - 0.3061666786670685 Conv_0_Relu_ts[1,0]
   + 0.8854666948318481 Conv_0_Relu_ts[1,1]
   - 1.108363628387451 Conv_0_Relu_ts[1,2]
   - 0.0485447347164154 Conv_0_Relu_ts[1,3]
   - 0.3054571747779846 Conv_0_Relu_ts[1,4]
   - 1.666001558303833 Conv_0_Relu_ts[1,5]
   + 1.575538277626038 Conv_0_Relu_ts[1,6]
   - 0.8786032199859619 Conv_0_Relu_ts[1,7]
   + 0.1696695387363434 Conv_0_Relu_ts[1,8]
   - 0.2058112174272537 Conv_0_Relu_ts[1,9]
   - 0.8619717359542847 Conv_0_Relu_ts[1,10]
   + 1.268710613250732 Conv_0_Relu_ts[1,11]
   + 0.1005195453763008 Conv_0_Relu_ts[1,12]
   - 0.00627355137839913 Conv_0_Relu_ts[1,13]
   - 0.6727985143661499 Conv_0_Relu_ts[1,14]
   - 0.6082888841629028 Conv_0_Relu_ts[1,15]
   + 0.0746944323182106 Conv_1_aggregated_features[1,0]
   - 0.5317568182945251 Conv_1_aggregated_features[1,1]
   + 0.4405037760734558 Conv_1_aggregated_features[1,2]
   - 0.1488203704357147 Conv_1_aggregated_features[1,3]
   + 2.01956033706665 Conv_1_aggregated_features[1,4]
   + 0.8369019031524658 Conv_1_aggregated_features[1,5]
   + 0.1561367213726044 Conv_1_aggregated_features[1,6]
   + 0.1698544919490814 Conv_1_aggregated_features[1,7]
   - 0.0660476386547089 Conv_1_aggregated_features[1,8]
   + 1.570046067237854 Conv_1_aggregated_features[1,9]
   + 0.5540977120399475 Conv_1_aggregated_features[1,10]
   - 1.275410532951355 Conv_1_aggregated_features[1,11]
   - 0.092900924384594 Conv_1_aggregated_features[1,12]
   + 3.287018537521362 Conv_1_aggregated_features[1,13]
   + 2.392175912857056 Conv_1_aggregated_features[1,14]
   + 0.3933185040950775 Conv_1_aggregated_features[1,15] + Conv_1_t[1,0]
   = -2.192610263824463
 Conv_1_output_constraint[1,1]: 0.5202509164810181 Conv_0_Relu_ts[1,0]
   + 0.6686695218086243 Conv_0_Relu_ts[1,1]
   - 0.4720649719238281 Conv_0_Relu_ts[1,2]
   + 0.1026173532009125 Conv_0_Relu_ts[1,3]
   + 0.2593795359134674 Conv_0_Relu_ts[1,4]
   - 2.458898544311523 Conv_0_Relu_ts[1,5]
   - 0.5632196664810181 Conv_0_Relu_ts[1,6]
   - 1.791601657867432 Conv_0_Relu_ts[1,7]
   - 0.1268838942050934 Conv_0_Relu_ts[1,8]
   + 0.7120329737663269 Conv_0_Relu_ts[1,9]
   - 0.5212016701698303 Conv_0_Relu_ts[1,10]
   + 0.6136131286621094 Conv_0_Relu_ts[1,11]
   + 1.563590168952942 Conv_0_Relu_ts[1,12]
   - 0.3512922525405884 Conv_0_Relu_ts[1,13]
   + 0.5421575903892517 Conv_0_Relu_ts[1,14]
   + 0.7331249117851257 Conv_0_Relu_ts[1,15]
   + 0.0113700022920966 Conv_1_aggregated_features[1,0]
   + 2.169116258621216 Conv_1_aggregated_features[1,1]
   + 0.00623909616842866 Conv_1_aggregated_features[1,2]
   + 0.0937856733798981 Conv_1_aggregated_features[1,3]
   + 0.8328568339347839 Conv_1_aggregated_features[1,4]
   + 0.383074939250946 Conv_1_aggregated_features[1,5]
   - 0.1267539262771606 Conv_1_aggregated_features[1,6]
   + 0.6484018564224243 Conv_1_aggregated_features[1,7]
   + 0.1870926916599274 Conv_1_aggregated_features[1,8]
   - 0.4116691052913666 Conv_1_aggregated_features[1,9]
   + 0.1439376026391983 Conv_1_aggregated_features[1,10]
   + 1.05384635925293 Conv_1_aggregated_features[1,11]
   + 1.155439853668213 Conv_1_aggregated_features[1,12]
   - 0.2060132175683975 Conv_1_aggregated_features[1,13]
   + 0.6003169417381287 Conv_1_aggregated_features[1,14]
   + 0.1458597332239151 Conv_1_aggregated_features[1,15] + Conv_1_t[1,1]
   = -0.1077260375022888
 Conv_1_output_constraint[1,2]: 0.4269110858440399 Conv_0_Relu_ts[1,0]
   - 1.097789287567139 Conv_0_Relu_ts[1,1]
   - 0.3380239307880402 Conv_0_Relu_ts[1,2]
   + 0.23756343126297 Conv_0_Relu_ts[1,3]
   - 0.337738424539566 Conv_0_Relu_ts[1,4]
   - 1.407688736915588 Conv_0_Relu_ts[1,5]
   - 0.7854894995689392 Conv_0_Relu_ts[1,6]
   - 1.080899715423584 Conv_0_Relu_ts[1,7]
   + 0.0233876463025808 Conv_0_Relu_ts[1,8]
   + 1.849825739860535 Conv_0_Relu_ts[1,9]
   + 0.4580322206020355 Conv_0_Relu_ts[1,10]
   - 0.819361686706543 Conv_0_Relu_ts[1,11]
   - 0.0822571739554405 Conv_0_Relu_ts[1,12]
   - 0.1501271426677704 Conv_0_Relu_ts[1,13]
   + 0.1831801384687424 Conv_0_Relu_ts[1,14]
   + 0.00308510870672762 Conv_0_Relu_ts[1,15]
   - 0.2153502702713013 Conv_1_aggregated_features[1,0]
   + 3.817557334899902 Conv_1_aggregated_features[1,1]
   - 0.3402683734893799 Conv_1_aggregated_features[1,2]
   + 0.0650523602962494 Conv_1_aggregated_features[1,3]
   - 0.4435569643974304 Conv_1_aggregated_features[1,4]
   + 0.045248981565237 Conv_1_aggregated_features[1,5]
   + 0.844337522983551 Conv_1_aggregated_features[1,6]
   - 0.0691490024328232 Conv_1_aggregated_features[1,7]
   + 0.0830505788326263 Conv_1_aggregated_features[1,8]
   - 0.3022302687168121 Conv_1_aggregated_features[1,9]
   - 0.611566960811615 Conv_1_aggregated_features[1,10]
   + 3.582128524780273 Conv_1_aggregated_features[1,11]
   + 0.1314867585897446 Conv_1_aggregated_features[1,12]
   + 0.5333523154258728 Conv_1_aggregated_features[1,13]
   - 0.1983102858066559 Conv_1_aggregated_features[1,14]
   + 0.1628565639257431 Conv_1_aggregated_features[1,15] + Conv_1_t[1,2]
   = -0.4443503022193909
 Conv_1_output_constraint[1,3]: 0.066770151257515 Conv_0_Relu_ts[1,0]
   + 0.3136637210845947 Conv_0_Relu_ts[1,1]
   + 0.2078424692153931 Conv_0_Relu_ts[1,2]
   - 0.1673517823219299 Conv_0_Relu_ts[1,3]
   + 1.022224068641663 Conv_0_Relu_ts[1,4]
   - 0.1723377108573914 Conv_0_Relu_ts[1,5]
   + 0.3032898306846619 Conv_0_Relu_ts[1,6]
   + 0.0841449126601219 Conv_0_Relu_ts[1,7]
   + 0.0921086072921753 Conv_0_Relu_ts[1,8]
   - 0.022469462826848 Conv_0_Relu_ts[1,9]
   - 0.1657154262065887 Conv_0_Relu_ts[1,10]
   + 1.007031679153442 Conv_0_Relu_ts[1,11]
   + 0.0208180602639914 Conv_0_Relu_ts[1,12]
   + 0.3933880031108856 Conv_0_Relu_ts[1,13]
   + 0.3077191412448883 Conv_0_Relu_ts[1,14]
   + 0.0400979965925217 Conv_0_Relu_ts[1,15]
   + 0.0230858232825994 Conv_1_aggregated_features[1,0]
   + 0.3792195022106171 Conv_1_aggregated_features[1,1]
   + 0.1572305560112 Conv_1_aggregated_features[1,2]
   - 0.0165807008743286 Conv_1_aggregated_features[1,3]
   + 0.5649795532226562 Conv_1_aggregated_features[1,4]
   + 0.6253551840782166 Conv_1_aggregated_features[1,5]
   + 0.3968201577663422 Conv_1_aggregated_features[1,6]
   + 0.2728661298751831 Conv_1_aggregated_features[1,7]
   - 0.210608571767807 Conv_1_aggregated_features[1,8]
   + 0.094737209379673 Conv_1_aggregated_features[1,9]
   - 0.1354089975357056 Conv_1_aggregated_features[1,10]
   + 0.9330839514732361 Conv_1_aggregated_features[1,11]
   + 0.1677136421203613 Conv_1_aggregated_features[1,12]
   + 0.6573846936225891 Conv_1_aggregated_features[1,13]
   + 0.044663343578577 Conv_1_aggregated_features[1,14]
   - 0.0132347755134106 Conv_1_aggregated_features[1,15] + Conv_1_t[1,3]
   = -0.327608048915863
 Conv_1_output_constraint[1,4]: - 0.0720538049936295 Conv_0_Relu_ts[1,0]
   + 3.730369091033936 Conv_0_Relu_ts[1,1]
   - 0.8118388652801514 Conv_0_Relu_ts[1,2]
   - 0.174347460269928 Conv_0_Relu_ts[1,3]
   - 3.084089279174805 Conv_0_Relu_ts[1,4]
   - 5.146248340606689 Conv_0_Relu_ts[1,5]
   - 2.440467357635498 Conv_0_Relu_ts[1,6]
   - 3.765322923660278 Conv_0_Relu_ts[1,7]
   + 0.153628796339035 Conv_0_Relu_ts[1,8]
   + 0.1013353765010834 Conv_0_Relu_ts[1,9]
   - 0.5803871154785156 Conv_0_Relu_ts[1,10]
   - 2.657181739807129 Conv_0_Relu_ts[1,11]
   + 0.6840037703514099 Conv_0_Relu_ts[1,12]
   - 1.184461355209351 Conv_0_Relu_ts[1,13]
   - 0.3772841989994049 Conv_0_Relu_ts[1,14]
   + 0.5856359601020813 Conv_0_Relu_ts[1,15]
   + 0.2062403708696365 Conv_1_aggregated_features[1,0]
   + 1.470675468444824 Conv_1_aggregated_features[1,1]
   + 0.3523006737232208 Conv_1_aggregated_features[1,2]
   - 0.1351108849048615 Conv_1_aggregated_features[1,3]
   + 0.8932182788848877 Conv_1_aggregated_features[1,4]
   + 1.194340229034424 Conv_1_aggregated_features[1,5]
   + 0.0163087919354439 Conv_1_aggregated_features[1,6]
   - 0.5082771182060242 Conv_1_aggregated_features[1,7]
   - 0.0201514065265656 Conv_1_aggregated_features[1,8]
   + 0.9045064449310303 Conv_1_aggregated_features[1,9]
   + 0.7243757843971252 Conv_1_aggregated_features[1,10]
   + 0.5823414921760559 Conv_1_aggregated_features[1,11]
   + 0.0738087296485901 Conv_1_aggregated_features[1,12]
   - 0.3080458641052246 Conv_1_aggregated_features[1,13]
   + 0.9457569122314453 Conv_1_aggregated_features[1,14]
   + 0.4423830211162567 Conv_1_aggregated_features[1,15] + Conv_1_t[1,4]
   = -0.2138544023036957
 Conv_1_output_constraint[1,5]: 0.2116706073284149 Conv_0_Relu_ts[1,0]
   + 0.2319677472114563 Conv_0_Relu_ts[1,1]
   + 0.1967353820800781 Conv_0_Relu_ts[1,2]
   - 0.1974936425685883 Conv_0_Relu_ts[1,3]
   + 0.087493434548378 Conv_0_Relu_ts[1,4]
   + 0.0226023308932781 Conv_0_Relu_ts[1,5]
   - 0.1455883532762527 Conv_0_Relu_ts[1,6]
   - 0.1098284423351288 Conv_0_Relu_ts[1,7]
   + 0.0375414304435253 Conv_0_Relu_ts[1,8]
   + 0.2209681868553162 Conv_0_Relu_ts[1,9]
   + 0.0180207490921021 Conv_0_Relu_ts[1,10]
   - 0.0944192707538605 Conv_0_Relu_ts[1,11]
   + 0.1318628787994385 Conv_0_Relu_ts[1,12]
   + 0.1718421876430511 Conv_0_Relu_ts[1,13]
   - 0.0992768108844757 Conv_0_Relu_ts[1,14]
   + 0.2220577001571655 Conv_0_Relu_ts[1,15]
   + 0.0500434041023254 Conv_1_aggregated_features[1,0]
   + 0.2417693138122559 Conv_1_aggregated_features[1,1]
   + 0.1884889900684357 Conv_1_aggregated_features[1,2]
   - 0.0459566116333008 Conv_1_aggregated_features[1,3]
   - 0.1095028519630432 Conv_1_aggregated_features[1,4]
   - 0.1494371891021729 Conv_1_aggregated_features[1,5]
   - 0.0601528584957123 Conv_1_aggregated_features[1,6]
   - 0.2431226968765259 Conv_1_aggregated_features[1,7]
   + 0.0709281861782074 Conv_1_aggregated_features[1,8]
   + 0.1859873235225677 Conv_1_aggregated_features[1,9]
   - 0.0025022029876709 Conv_1_aggregated_features[1,10]
   + 0.1165908575057983 Conv_1_aggregated_features[1,11]
   - 0.0946725904941559 Conv_1_aggregated_features[1,12]
   + 0.1679102778434753 Conv_1_aggregated_features[1,13]
   + 0.0888539850711823 Conv_1_aggregated_features[1,14]
   + 0.0481749773025513 Conv_1_aggregated_features[1,15] + Conv_1_t[1,5]
   = -0.2017861604690552
 Conv_1_output_constraint[1,6]: - 0.5161762237548828 Conv_0_Relu_ts[1,0]
   - 0.9039249420166016 Conv_0_Relu_ts[1,1]
   - 0.5091935992240906 Conv_0_Relu_ts[1,2]
   + 0.177293062210083 Conv_0_Relu_ts[1,3]
   + 1.534894466400146 Conv_0_Relu_ts[1,4]
   + 0.3099820613861084 Conv_0_Relu_ts[1,5]
   + 1.335353136062622 Conv_0_Relu_ts[1,6]
   - 0.3965733051300049 Conv_0_Relu_ts[1,7]
   + 0.0226722583174706 Conv_0_Relu_ts[1,8]
   - 0.0153551828116179 Conv_0_Relu_ts[1,9]
   - 0.188335582613945 Conv_0_Relu_ts[1,10]
   + 0.1465377509593964 Conv_0_Relu_ts[1,11]
   - 0.4378408491611481 Conv_0_Relu_ts[1,12]
   + 0.5950998067855835 Conv_0_Relu_ts[1,13]
   - 0.2773410975933075 Conv_0_Relu_ts[1,14]
   - 0.3855878710746765 Conv_0_Relu_ts[1,15]
   + 0.0145458607003093 Conv_1_aggregated_features[1,0]
   - 1.349490284919739 Conv_1_aggregated_features[1,1]
   - 0.6738612055778503 Conv_1_aggregated_features[1,2]
   - 0.1977581083774567 Conv_1_aggregated_features[1,3]
   + 3.616869688034058 Conv_1_aggregated_features[1,4]
   - 0.5846084952354431 Conv_1_aggregated_features[1,5]
   + 0.3911246955394745 Conv_1_aggregated_features[1,6]
   - 0.2603789269924164 Conv_1_aggregated_features[1,7]
   + 0.0530429184436798 Conv_1_aggregated_features[1,8]
   + 0.7039647698402405 Conv_1_aggregated_features[1,9]
   - 0.4948081970214844 Conv_1_aggregated_features[1,10]
   - 1.33747124671936 Conv_1_aggregated_features[1,11]
   + 0.4521820247173309 Conv_1_aggregated_features[1,12]
   + 2.879487276077271 Conv_1_aggregated_features[1,13]
   + 0.6494181156158447 Conv_1_aggregated_features[1,14]
   - 0.0441654324531555 Conv_1_aggregated_features[1,15] + Conv_1_t[1,6]
   = -1.141431212425232
 Conv_1_output_constraint[1,7]: 0.1232976838946342 Conv_0_Relu_ts[1,0]
   + 0.7059634327888489 Conv_0_Relu_ts[1,1]
   + 0.0986589044332504 Conv_0_Relu_ts[1,2]
   - 0.1159188747406006 Conv_0_Relu_ts[1,3]
   + 0.1207201555371284 Conv_0_Relu_ts[1,4]
   + 0.4950260519981384 Conv_0_Relu_ts[1,5]
   - 0.1985135078430176 Conv_0_Relu_ts[1,6]
   + 0.3846564292907715 Conv_0_Relu_ts[1,7]
   - 0.2091868817806244 Conv_0_Relu_ts[1,8]
   + 0.0282646510750055 Conv_0_Relu_ts[1,9]
   + 0.00616153934970498 Conv_0_Relu_ts[1,10]
   - 0.2992357313632965 Conv_0_Relu_ts[1,11]
   + 0.6717308759689331 Conv_0_Relu_ts[1,12]
   + 0.3884595632553101 Conv_0_Relu_ts[1,13]
   + 0.1088093742728233 Conv_0_Relu_ts[1,14]
   + 0.1363743990659714 Conv_0_Relu_ts[1,15]
   + 0.2181874513626099 Conv_1_aggregated_features[1,0]
   + 0.1884859055280685 Conv_1_aggregated_features[1,1]
   + 0.1986039727926254 Conv_1_aggregated_features[1,2]
   - 0.0569350719451904 Conv_1_aggregated_features[1,3]
   + 0.4957895576953888 Conv_1_aggregated_features[1,4]
   - 0.2215360552072525 Conv_1_aggregated_features[1,5]
   - 0.3126825392246246 Conv_1_aggregated_features[1,6]
   + 0.2129529714584351 Conv_1_aggregated_features[1,7]
   + 0.1905750632286072 Conv_1_aggregated_features[1,8]
   + 0.3835417926311493 Conv_1_aggregated_features[1,9]
   + 0.0515264272689819 Conv_1_aggregated_features[1,10]
   + 0.2774861454963684 Conv_1_aggregated_features[1,11]
   + 0.1720549166202545 Conv_1_aggregated_features[1,12]
   + 0.2042496353387833 Conv_1_aggregated_features[1,13]
   + 0.1717791706323624 Conv_1_aggregated_features[1,14]
   - 0.0635386854410172 Conv_1_aggregated_features[1,15] + Conv_1_t[1,7]
   = -0.1961010098457336
 Conv_1_output_constraint[1,8]: - 0.0984712317585945 Conv_0_Relu_ts[1,0]
   - 0.8972246050834656 Conv_0_Relu_ts[1,1]
   - 0.1342054009437561 Conv_0_Relu_ts[1,2]
   + 0.0934775173664093 Conv_0_Relu_ts[1,3]
   + 3.929172039031982 Conv_0_Relu_ts[1,4]
   - 1.699812531471252 Conv_0_Relu_ts[1,5]
   + 0.4780299663543701 Conv_0_Relu_ts[1,6]
   + 0.1916014850139618 Conv_0_Relu_ts[1,7]
   - 0.0282735247164965 Conv_0_Relu_ts[1,8]
   - 0.10334312915802 Conv_0_Relu_ts[1,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[1,10]
   - 0.9000989198684692 Conv_0_Relu_ts[1,11]
   - 0.3381493389606476 Conv_0_Relu_ts[1,12]
   + 0.2327869236469269 Conv_0_Relu_ts[1,13]
   + 0.6162272691726685 Conv_0_Relu_ts[1,14]
   - 0.1045716181397438 Conv_0_Relu_ts[1,15]
   - 0.0895406752824783 Conv_1_aggregated_features[1,0]
   - 0.0733596011996269 Conv_1_aggregated_features[1,1]
   - 0.6064443588256836 Conv_1_aggregated_features[1,2]
   + 0.1685465276241302 Conv_1_aggregated_features[1,3]
   + 2.521227836608887 Conv_1_aggregated_features[1,4]
   + 0.8868938088417053 Conv_1_aggregated_features[1,5]
   - 0.3730944097042084 Conv_1_aggregated_features[1,6]
   + 0.4433300495147705 Conv_1_aggregated_features[1,7]
   + 0.2465403079986572 Conv_1_aggregated_features[1,8]
   + 0.1199371814727783 Conv_1_aggregated_features[1,9]
   - 0.427333652973175 Conv_1_aggregated_features[1,10]
   - 1.286244511604309 Conv_1_aggregated_features[1,11]
   + 0.4271916449069977 Conv_1_aggregated_features[1,12]
   + 1.475327014923096 Conv_1_aggregated_features[1,13]
   - 0.2351342439651489 Conv_1_aggregated_features[1,14]
   - 0.1273849010467529 Conv_1_aggregated_features[1,15] + Conv_1_t[1,8]
   = -1.303357601165771
 Conv_1_output_constraint[1,9]: 0.1534582376480103 Conv_0_Relu_ts[1,0]
   + 1.456407189369202 Conv_0_Relu_ts[1,1]
   - 0.0228729248046875 Conv_0_Relu_ts[1,2]
   + 0.1227743625640869 Conv_0_Relu_ts[1,3]
   + 0.7793847918510437 Conv_0_Relu_ts[1,4]
   - 0.2757052779197693 Conv_0_Relu_ts[1,5]
   + 0.6606881022453308 Conv_0_Relu_ts[1,6]
   - 1.107266664505005 Conv_0_Relu_ts[1,7]
   + 0.1723826229572296 Conv_0_Relu_ts[1,8]
   - 0.5540673732757568 Conv_0_Relu_ts[1,9]
   + 0.3133971095085144 Conv_0_Relu_ts[1,10]
   + 0.9318023324012756 Conv_0_Relu_ts[1,11]
   + 0.5855908989906311 Conv_0_Relu_ts[1,12]
   - 0.2116718590259552 Conv_0_Relu_ts[1,13]
   - 0.3693177700042725 Conv_0_Relu_ts[1,14]
   - 0.0642165616154671 Conv_0_Relu_ts[1,15]
   + 0.1819062232971191 Conv_1_aggregated_features[1,0]
   + 1.311912417411804 Conv_1_aggregated_features[1,1]
   + 0.3460695445537567 Conv_1_aggregated_features[1,2]
   + 0.1086616516113281 Conv_1_aggregated_features[1,3]
   - 0.6099227070808411 Conv_1_aggregated_features[1,4]
   + 0.3786990940570831 Conv_1_aggregated_features[1,5]
   + 0.9473716020584106 Conv_1_aggregated_features[1,6]
   + 0.2196423262357712 Conv_1_aggregated_features[1,7]
   + 0.0830722749233246 Conv_1_aggregated_features[1,8]
   - 0.4822347462177277 Conv_1_aggregated_features[1,9]
   + 0.5108910202980042 Conv_1_aggregated_features[1,10]
   + 3.214276790618896 Conv_1_aggregated_features[1,11]
   - 0.7224436402320862 Conv_1_aggregated_features[1,12]
   + 0.2678262591362 Conv_1_aggregated_features[1,13]
   - 0.8807565569877625 Conv_1_aggregated_features[1,14]
   - 0.3268090188503265 Conv_1_aggregated_features[1,15] + Conv_1_t[1,9]
   = -0.8283404111862183
 Conv_1_output_constraint[1,10]: 0.3653882741928101 Conv_0_Relu_ts[1,0]
   - 0.8208209872245789 Conv_0_Relu_ts[1,1]
   + 0.3088222146034241 Conv_0_Relu_ts[1,2]
   + 0.1173163652420044 Conv_0_Relu_ts[1,3]
   + 0.658578634262085 Conv_0_Relu_ts[1,4]
   + 0.584019660949707 Conv_0_Relu_ts[1,5]
   - 0.8654046058654785 Conv_0_Relu_ts[1,6]
   + 0.3697350919246674 Conv_0_Relu_ts[1,7]
   - 0.2274059355258942 Conv_0_Relu_ts[1,8]
   + 1.197938442230225 Conv_0_Relu_ts[1,9]
   + 0.2446341812610626 Conv_0_Relu_ts[1,10]
   - 0.5676743984222412 Conv_0_Relu_ts[1,11]
   - 0.3123646676540375 Conv_0_Relu_ts[1,12]
   + 3.181503534317017 Conv_0_Relu_ts[1,13]
   + 0.6926240921020508 Conv_0_Relu_ts[1,14]
   + 0.0886927545070648 Conv_0_Relu_ts[1,15]
   - 0.2005636394023895 Conv_1_aggregated_features[1,0]
   + 1.403087973594666 Conv_1_aggregated_features[1,1]
   + 0.0523448958992958 Conv_1_aggregated_features[1,2]
   + 0.2411543428897858 Conv_1_aggregated_features[1,3]
   + 0.5953171253204346 Conv_1_aggregated_features[1,4]
   - 0.8278964757919312 Conv_1_aggregated_features[1,5]
   + 0.2499217540025711 Conv_1_aggregated_features[1,6]
   - 0.7393644452095032 Conv_1_aggregated_features[1,7]
   + 0.0902875959873199 Conv_1_aggregated_features[1,8]
   + 0.1488711386919022 Conv_1_aggregated_features[1,9]
   - 0.0303693022578955 Conv_1_aggregated_features[1,10]
   + 2.024244070053101 Conv_1_aggregated_features[1,11]
   + 0.495431512594223 Conv_1_aggregated_features[1,12]
   + 0.3463623225688934 Conv_1_aggregated_features[1,13]
   - 0.0763808861374855 Conv_1_aggregated_features[1,14]
   + 0.1597562879323959 Conv_1_aggregated_features[1,15] + Conv_1_t[1,10]
   = -0.5063248872756958
 Conv_1_output_constraint[1,11]: - 0.5185666680335999 Conv_0_Relu_ts[1,0]
   + 3.656918525695801 Conv_0_Relu_ts[1,1]
   - 0.7082476615905762 Conv_0_Relu_ts[1,2]
   + 0.1922211349010468 Conv_0_Relu_ts[1,3]
   - 0.7149283289909363 Conv_0_Relu_ts[1,4]
   - 3.255658626556396 Conv_0_Relu_ts[1,5]
   + 2.464392185211182 Conv_0_Relu_ts[1,6]
   - 0.9477378129959106 Conv_0_Relu_ts[1,7]
   + 0.168237030506134 Conv_0_Relu_ts[1,8]
   - 1.177640557289124 Conv_0_Relu_ts[1,9]
   - 0.7062549591064453 Conv_0_Relu_ts[1,10]
   + 1.453806400299072 Conv_0_Relu_ts[1,11]
   + 0.2612711191177368 Conv_0_Relu_ts[1,12]
   + 0.2999382615089417 Conv_0_Relu_ts[1,13]
   - 1.255579590797424 Conv_0_Relu_ts[1,14]
   - 0.4265820384025574 Conv_0_Relu_ts[1,15]
   + 0.3379200994968414 Conv_1_aggregated_features[1,0]
   - 1.33229672908783 Conv_1_aggregated_features[1,1]
   - 0.1594242304563522 Conv_1_aggregated_features[1,2]
   + 0.2298935651779175 Conv_1_aggregated_features[1,3]
   - 1.311494588851929 Conv_1_aggregated_features[1,4]
   + 3.605161666870117 Conv_1_aggregated_features[1,5]
   - 0.3176752626895905 Conv_1_aggregated_features[1,6]
   + 0.3926282525062561 Conv_1_aggregated_features[1,7]
   + 0.1479305028915405 Conv_1_aggregated_features[1,8]
   - 0.0372937023639679 Conv_1_aggregated_features[1,9]
   + 0.3455427587032318 Conv_1_aggregated_features[1,10]
   - 2.002299070358276 Conv_1_aggregated_features[1,11]
   - 0.9904713034629822 Conv_1_aggregated_features[1,12]
   + 0.0751789212226868 Conv_1_aggregated_features[1,13]
   - 0.5166233777999878 Conv_1_aggregated_features[1,14]
   + 0.0510688796639442 Conv_1_aggregated_features[1,15] + Conv_1_t[1,11]
   = -1.480785727500916
 Conv_1_output_constraint[1,12]: 0.3080185949802399 Conv_0_Relu_ts[1,0]
   + 0.0994751453399658 Conv_0_Relu_ts[1,1]
   + 0.4488444924354553 Conv_0_Relu_ts[1,2]
   + 0.0305773317813873 Conv_0_Relu_ts[1,3]
   - 1.673393130302429 Conv_0_Relu_ts[1,4]
   + 0.2824500501155853 Conv_0_Relu_ts[1,5]
   - 1.192709684371948 Conv_0_Relu_ts[1,6]
   + 1.454243063926697 Conv_0_Relu_ts[1,7]
   + 0.1514801830053329 Conv_0_Relu_ts[1,8]
   + 1.026351928710938 Conv_0_Relu_ts[1,9]
   + 0.1183762699365616 Conv_0_Relu_ts[1,10]
   + 0.352224588394165 Conv_0_Relu_ts[1,11]
   + 1.090276837348938 Conv_0_Relu_ts[1,12]
   - 2.053535461425781 Conv_0_Relu_ts[1,13]
   + 2.538641452789307 Conv_0_Relu_ts[1,14]
   + 0.6387560963630676 Conv_0_Relu_ts[1,15]
   - 0.1990496516227722 Conv_1_aggregated_features[1,0]
   + 1.247365713119507 Conv_1_aggregated_features[1,1]
   + 0.2213435918092728 Conv_1_aggregated_features[1,2]
   + 0.2398849725723267 Conv_1_aggregated_features[1,3]
   - 2.001479864120483 Conv_1_aggregated_features[1,4]
   - 1.084800243377686 Conv_1_aggregated_features[1,5]
   + 0.4287458956241608 Conv_1_aggregated_features[1,6]
   - 0.8179780840873718 Conv_1_aggregated_features[1,7]
   - 0.0883614718914032 Conv_1_aggregated_features[1,8]
   - 0.4398066997528076 Conv_1_aggregated_features[1,9]
   + 0.2760448455810547 Conv_1_aggregated_features[1,10]
   + 1.944519639015198 Conv_1_aggregated_features[1,11]
   + 1.041437149047852 Conv_1_aggregated_features[1,12]
   - 1.682050704956055 Conv_1_aggregated_features[1,13]
   - 0.1825132220983505 Conv_1_aggregated_features[1,14]
   + 0.2881160080432892 Conv_1_aggregated_features[1,15] + Conv_1_t[1,12]
   = 2.457676649093628
 Conv_1_output_constraint[1,13]: 0.2853958606719971 Conv_0_Relu_ts[1,0]
   + 0.0445350483059883 Conv_0_Relu_ts[1,1]
   - 0.0597315393388271 Conv_0_Relu_ts[1,2]
   + 0.2403226792812347 Conv_0_Relu_ts[1,3]
   + 0.5103238224983215 Conv_0_Relu_ts[1,4]
   - 0.1369927227497101 Conv_0_Relu_ts[1,5]
   + 0.5091087818145752 Conv_0_Relu_ts[1,6]
   - 0.0576926171779633 Conv_0_Relu_ts[1,7]
   - 0.00701320171356201 Conv_0_Relu_ts[1,8]
   + 0.5782948136329651 Conv_0_Relu_ts[1,9]
   - 0.1188536584377289 Conv_0_Relu_ts[1,10]
   - 0.078026682138443 Conv_0_Relu_ts[1,11]
   + 0.1965750753879547 Conv_0_Relu_ts[1,12]
   + 0.4982857704162598 Conv_0_Relu_ts[1,13]
   - 0.0995257869362831 Conv_0_Relu_ts[1,14]
   + 0.0712818428874016 Conv_0_Relu_ts[1,15]
   + 0.2628078460693359 Conv_1_aggregated_features[1,0]
   + 0.6577805280685425 Conv_1_aggregated_features[1,1]
   - 0.1415485441684723 Conv_1_aggregated_features[1,2]
   + 0.1597791016101837 Conv_1_aggregated_features[1,3]
   + 0.3969745337963104 Conv_1_aggregated_features[1,4]
   - 0.3941676616668701 Conv_1_aggregated_features[1,5]
   + 0.3484204709529877 Conv_1_aggregated_features[1,6]
   + 0.2601790428161621 Conv_1_aggregated_features[1,7]
   - 0.0652478933334351 Conv_1_aggregated_features[1,8]
   + 0.3492280542850494 Conv_1_aggregated_features[1,9]
   + 0.1917867064476013 Conv_1_aggregated_features[1,10]
   + 0.2439181953668594 Conv_1_aggregated_features[1,11]
   + 0.1009163707494736 Conv_1_aggregated_features[1,12]
   + 0.2350947707891464 Conv_1_aggregated_features[1,13]
   + 0.2627962231636047 Conv_1_aggregated_features[1,14]
   - 0.0959920063614845 Conv_1_aggregated_features[1,15] + Conv_1_t[1,13]
   = -0.4268372058868408
 Conv_1_output_constraint[1,14]: - 0.2448655962944031 Conv_0_Relu_ts[1,0]
   + 0.2688764035701752 Conv_0_Relu_ts[1,1]
   + 0.0118266940116882 Conv_0_Relu_ts[1,2]
   - 0.0670525431632996 Conv_0_Relu_ts[1,3]
   + 0.1511579602956772 Conv_0_Relu_ts[1,4]
   - 0.4796198606491089 Conv_0_Relu_ts[1,5]
   - 0.1699391901493073 Conv_0_Relu_ts[1,6]
   + 0.2647148370742798 Conv_0_Relu_ts[1,7]
   + 0.137117475271225 Conv_0_Relu_ts[1,8]
   + 0.1737731248140335 Conv_0_Relu_ts[1,9]
   - 0.1610838174819946 Conv_0_Relu_ts[1,10]
   - 0.6311220526695251 Conv_0_Relu_ts[1,11]
   + 0.6263757348060608 Conv_0_Relu_ts[1,12]
   - 0.4823384284973145 Conv_0_Relu_ts[1,13]
   + 0.2192765176296234 Conv_0_Relu_ts[1,14]
   + 0.1739131808280945 Conv_0_Relu_ts[1,15]
   + 0.0443203188478947 Conv_1_aggregated_features[1,0]
   + 0.7668815851211548 Conv_1_aggregated_features[1,1]
   + 0.248365581035614 Conv_1_aggregated_features[1,2]
   - 0.2433404624462128 Conv_1_aggregated_features[1,3]
   + 0.2770798802375793 Conv_1_aggregated_features[1,4]
   - 1.224791765213013 Conv_1_aggregated_features[1,5]
   - 0.1892879456281662 Conv_1_aggregated_features[1,6]
   + 0.1141589805483818 Conv_1_aggregated_features[1,7]
   + 0.2022489905357361 Conv_1_aggregated_features[1,8]
   - 0.0391396470367908 Conv_1_aggregated_features[1,9]
   - 0.0935317650437355 Conv_1_aggregated_features[1,10]
   - 0.5407891273498535 Conv_1_aggregated_features[1,11]
   + 0.3130630552768707 Conv_1_aggregated_features[1,12]
   + 0.3410882651805878 Conv_1_aggregated_features[1,13]
   - 1.855417490005493 Conv_1_aggregated_features[1,14]
   + 0.247613787651062 Conv_1_aggregated_features[1,15] + Conv_1_t[1,14]
   = -0.2306040376424789
 Conv_1_output_constraint[1,15]: 0.043729841709137 Conv_0_Relu_ts[1,0]
   + 0.045797374099493 Conv_0_Relu_ts[1,1]
   - 0.1762047111988068 Conv_0_Relu_ts[1,2]
   - 0.2225021123886108 Conv_0_Relu_ts[1,3]
   + 0.1344186365604401 Conv_0_Relu_ts[1,4]
   + 0.2975046932697296 Conv_0_Relu_ts[1,5]
   + 0.1135610193014145 Conv_0_Relu_ts[1,6]
   + 0.165085032582283 Conv_0_Relu_ts[1,7]
   + 0.0632660016417503 Conv_0_Relu_ts[1,8]
   + 0.2534148395061493 Conv_0_Relu_ts[1,9]
   + 0.1100017726421356 Conv_0_Relu_ts[1,10]
   + 0.045901358127594 Conv_0_Relu_ts[1,11]
   + 0.2102706432342529 Conv_0_Relu_ts[1,12]
   + 0.1767551600933075 Conv_0_Relu_ts[1,13]
   - 0.0864512398838997 Conv_0_Relu_ts[1,14]
   + 0.1616824418306351 Conv_0_Relu_ts[1,15]
   + 0.0113421082496643 Conv_1_aggregated_features[1,0]
   + 0.2387615889310837 Conv_1_aggregated_features[1,1]
   + 0.1763307154178619 Conv_1_aggregated_features[1,2]
   + 0.1442219316959381 Conv_1_aggregated_features[1,3]
   + 0.3573774099349976 Conv_1_aggregated_features[1,4]
   - 0.0406427308917046 Conv_1_aggregated_features[1,5]
   + 0.1155626475811005 Conv_1_aggregated_features[1,6]
   + 0.0186243671923876 Conv_1_aggregated_features[1,7]
   - 0.1389756500720978 Conv_1_aggregated_features[1,8]
   + 0.274962306022644 Conv_1_aggregated_features[1,9]
   + 0.0309249386191368 Conv_1_aggregated_features[1,10]
   - 0.1606657803058624 Conv_1_aggregated_features[1,11]
   + 0.2456246316432953 Conv_1_aggregated_features[1,12]
   + 0.4163841009140015 Conv_1_aggregated_features[1,13]
   + 0.29454505443573 Conv_1_aggregated_features[1,14]
   + 0.1688967496156693 Conv_1_aggregated_features[1,15] + Conv_1_t[1,15]
   = -0.112743653357029
 Conv_1_output_constraint[2,0]: - 0.3061666786670685 Conv_0_Relu_ts[2,0]
   + 0.8854666948318481 Conv_0_Relu_ts[2,1]
   - 1.108363628387451 Conv_0_Relu_ts[2,2]
   - 0.0485447347164154 Conv_0_Relu_ts[2,3]
   - 0.3054571747779846 Conv_0_Relu_ts[2,4]
   - 1.666001558303833 Conv_0_Relu_ts[2,5]
   + 1.575538277626038 Conv_0_Relu_ts[2,6]
   - 0.8786032199859619 Conv_0_Relu_ts[2,7]
   + 0.1696695387363434 Conv_0_Relu_ts[2,8]
   - 0.2058112174272537 Conv_0_Relu_ts[2,9]
   - 0.8619717359542847 Conv_0_Relu_ts[2,10]
   + 1.268710613250732 Conv_0_Relu_ts[2,11]
   + 0.1005195453763008 Conv_0_Relu_ts[2,12]
   - 0.00627355137839913 Conv_0_Relu_ts[2,13]
   - 0.6727985143661499 Conv_0_Relu_ts[2,14]
   - 0.6082888841629028 Conv_0_Relu_ts[2,15]
   + 0.0746944323182106 Conv_1_aggregated_features[2,0]
   - 0.5317568182945251 Conv_1_aggregated_features[2,1]
   + 0.4405037760734558 Conv_1_aggregated_features[2,2]
   - 0.1488203704357147 Conv_1_aggregated_features[2,3]
   + 2.01956033706665 Conv_1_aggregated_features[2,4]
   + 0.8369019031524658 Conv_1_aggregated_features[2,5]
   + 0.1561367213726044 Conv_1_aggregated_features[2,6]
   + 0.1698544919490814 Conv_1_aggregated_features[2,7]
   - 0.0660476386547089 Conv_1_aggregated_features[2,8]
   + 1.570046067237854 Conv_1_aggregated_features[2,9]
   + 0.5540977120399475 Conv_1_aggregated_features[2,10]
   - 1.275410532951355 Conv_1_aggregated_features[2,11]
   - 0.092900924384594 Conv_1_aggregated_features[2,12]
   + 3.287018537521362 Conv_1_aggregated_features[2,13]
   + 2.392175912857056 Conv_1_aggregated_features[2,14]
   + 0.3933185040950775 Conv_1_aggregated_features[2,15] + Conv_1_t[2,0]
   = -2.192610263824463
 Conv_1_output_constraint[2,1]: 0.5202509164810181 Conv_0_Relu_ts[2,0]
   + 0.6686695218086243 Conv_0_Relu_ts[2,1]
   - 0.4720649719238281 Conv_0_Relu_ts[2,2]
   + 0.1026173532009125 Conv_0_Relu_ts[2,3]
   + 0.2593795359134674 Conv_0_Relu_ts[2,4]
   - 2.458898544311523 Conv_0_Relu_ts[2,5]
   - 0.5632196664810181 Conv_0_Relu_ts[2,6]
   - 1.791601657867432 Conv_0_Relu_ts[2,7]
   - 0.1268838942050934 Conv_0_Relu_ts[2,8]
   + 0.7120329737663269 Conv_0_Relu_ts[2,9]
   - 0.5212016701698303 Conv_0_Relu_ts[2,10]
   + 0.6136131286621094 Conv_0_Relu_ts[2,11]
   + 1.563590168952942 Conv_0_Relu_ts[2,12]
   - 0.3512922525405884 Conv_0_Relu_ts[2,13]
   + 0.5421575903892517 Conv_0_Relu_ts[2,14]
   + 0.7331249117851257 Conv_0_Relu_ts[2,15]
   + 0.0113700022920966 Conv_1_aggregated_features[2,0]
   + 2.169116258621216 Conv_1_aggregated_features[2,1]
   + 0.00623909616842866 Conv_1_aggregated_features[2,2]
   + 0.0937856733798981 Conv_1_aggregated_features[2,3]
   + 0.8328568339347839 Conv_1_aggregated_features[2,4]
   + 0.383074939250946 Conv_1_aggregated_features[2,5]
   - 0.1267539262771606 Conv_1_aggregated_features[2,6]
   + 0.6484018564224243 Conv_1_aggregated_features[2,7]
   + 0.1870926916599274 Conv_1_aggregated_features[2,8]
   - 0.4116691052913666 Conv_1_aggregated_features[2,9]
   + 0.1439376026391983 Conv_1_aggregated_features[2,10]
   + 1.05384635925293 Conv_1_aggregated_features[2,11]
   + 1.155439853668213 Conv_1_aggregated_features[2,12]
   - 0.2060132175683975 Conv_1_aggregated_features[2,13]
   + 0.6003169417381287 Conv_1_aggregated_features[2,14]
   + 0.1458597332239151 Conv_1_aggregated_features[2,15] + Conv_1_t[2,1]
   = -0.1077260375022888
 Conv_1_output_constraint[2,2]: 0.4269110858440399 Conv_0_Relu_ts[2,0]
   - 1.097789287567139 Conv_0_Relu_ts[2,1]
   - 0.3380239307880402 Conv_0_Relu_ts[2,2]
   + 0.23756343126297 Conv_0_Relu_ts[2,3]
   - 0.337738424539566 Conv_0_Relu_ts[2,4]
   - 1.407688736915588 Conv_0_Relu_ts[2,5]
   - 0.7854894995689392 Conv_0_Relu_ts[2,6]
   - 1.080899715423584 Conv_0_Relu_ts[2,7]
   + 0.0233876463025808 Conv_0_Relu_ts[2,8]
   + 1.849825739860535 Conv_0_Relu_ts[2,9]
   + 0.4580322206020355 Conv_0_Relu_ts[2,10]
   - 0.819361686706543 Conv_0_Relu_ts[2,11]
   - 0.0822571739554405 Conv_0_Relu_ts[2,12]
   - 0.1501271426677704 Conv_0_Relu_ts[2,13]
   + 0.1831801384687424 Conv_0_Relu_ts[2,14]
   + 0.00308510870672762 Conv_0_Relu_ts[2,15]
   - 0.2153502702713013 Conv_1_aggregated_features[2,0]
   + 3.817557334899902 Conv_1_aggregated_features[2,1]
   - 0.3402683734893799 Conv_1_aggregated_features[2,2]
   + 0.0650523602962494 Conv_1_aggregated_features[2,3]
   - 0.4435569643974304 Conv_1_aggregated_features[2,4]
   + 0.045248981565237 Conv_1_aggregated_features[2,5]
   + 0.844337522983551 Conv_1_aggregated_features[2,6]
   - 0.0691490024328232 Conv_1_aggregated_features[2,7]
   + 0.0830505788326263 Conv_1_aggregated_features[2,8]
   - 0.3022302687168121 Conv_1_aggregated_features[2,9]
   - 0.611566960811615 Conv_1_aggregated_features[2,10]
   + 3.582128524780273 Conv_1_aggregated_features[2,11]
   + 0.1314867585897446 Conv_1_aggregated_features[2,12]
   + 0.5333523154258728 Conv_1_aggregated_features[2,13]
   - 0.1983102858066559 Conv_1_aggregated_features[2,14]
   + 0.1628565639257431 Conv_1_aggregated_features[2,15] + Conv_1_t[2,2]
   = -0.4443503022193909
 Conv_1_output_constraint[2,3]: 0.066770151257515 Conv_0_Relu_ts[2,0]
   + 0.3136637210845947 Conv_0_Relu_ts[2,1]
   + 0.2078424692153931 Conv_0_Relu_ts[2,2]
   - 0.1673517823219299 Conv_0_Relu_ts[2,3]
   + 1.022224068641663 Conv_0_Relu_ts[2,4]
   - 0.1723377108573914 Conv_0_Relu_ts[2,5]
   + 0.3032898306846619 Conv_0_Relu_ts[2,6]
   + 0.0841449126601219 Conv_0_Relu_ts[2,7]
   + 0.0921086072921753 Conv_0_Relu_ts[2,8]
   - 0.022469462826848 Conv_0_Relu_ts[2,9]
   - 0.1657154262065887 Conv_0_Relu_ts[2,10]
   + 1.007031679153442 Conv_0_Relu_ts[2,11]
   + 0.0208180602639914 Conv_0_Relu_ts[2,12]
   + 0.3933880031108856 Conv_0_Relu_ts[2,13]
   + 0.3077191412448883 Conv_0_Relu_ts[2,14]
   + 0.0400979965925217 Conv_0_Relu_ts[2,15]
   + 0.0230858232825994 Conv_1_aggregated_features[2,0]
   + 0.3792195022106171 Conv_1_aggregated_features[2,1]
   + 0.1572305560112 Conv_1_aggregated_features[2,2]
   - 0.0165807008743286 Conv_1_aggregated_features[2,3]
   + 0.5649795532226562 Conv_1_aggregated_features[2,4]
   + 0.6253551840782166 Conv_1_aggregated_features[2,5]
   + 0.3968201577663422 Conv_1_aggregated_features[2,6]
   + 0.2728661298751831 Conv_1_aggregated_features[2,7]
   - 0.210608571767807 Conv_1_aggregated_features[2,8]
   + 0.094737209379673 Conv_1_aggregated_features[2,9]
   - 0.1354089975357056 Conv_1_aggregated_features[2,10]
   + 0.9330839514732361 Conv_1_aggregated_features[2,11]
   + 0.1677136421203613 Conv_1_aggregated_features[2,12]
   + 0.6573846936225891 Conv_1_aggregated_features[2,13]
   + 0.044663343578577 Conv_1_aggregated_features[2,14]
   - 0.0132347755134106 Conv_1_aggregated_features[2,15] + Conv_1_t[2,3]
   = -0.327608048915863
 Conv_1_output_constraint[2,4]: - 0.0720538049936295 Conv_0_Relu_ts[2,0]
   + 3.730369091033936 Conv_0_Relu_ts[2,1]
   - 0.8118388652801514 Conv_0_Relu_ts[2,2]
   - 0.174347460269928 Conv_0_Relu_ts[2,3]
   - 3.084089279174805 Conv_0_Relu_ts[2,4]
   - 5.146248340606689 Conv_0_Relu_ts[2,5]
   - 2.440467357635498 Conv_0_Relu_ts[2,6]
   - 3.765322923660278 Conv_0_Relu_ts[2,7]
   + 0.153628796339035 Conv_0_Relu_ts[2,8]
   + 0.1013353765010834 Conv_0_Relu_ts[2,9]
   - 0.5803871154785156 Conv_0_Relu_ts[2,10]
   - 2.657181739807129 Conv_0_Relu_ts[2,11]
   + 0.6840037703514099 Conv_0_Relu_ts[2,12]
   - 1.184461355209351 Conv_0_Relu_ts[2,13]
   - 0.3772841989994049 Conv_0_Relu_ts[2,14]
   + 0.5856359601020813 Conv_0_Relu_ts[2,15]
   + 0.2062403708696365 Conv_1_aggregated_features[2,0]
   + 1.470675468444824 Conv_1_aggregated_features[2,1]
   + 0.3523006737232208 Conv_1_aggregated_features[2,2]
   - 0.1351108849048615 Conv_1_aggregated_features[2,3]
   + 0.8932182788848877 Conv_1_aggregated_features[2,4]
   + 1.194340229034424 Conv_1_aggregated_features[2,5]
   + 0.0163087919354439 Conv_1_aggregated_features[2,6]
   - 0.5082771182060242 Conv_1_aggregated_features[2,7]
   - 0.0201514065265656 Conv_1_aggregated_features[2,8]
   + 0.9045064449310303 Conv_1_aggregated_features[2,9]
   + 0.7243757843971252 Conv_1_aggregated_features[2,10]
   + 0.5823414921760559 Conv_1_aggregated_features[2,11]
   + 0.0738087296485901 Conv_1_aggregated_features[2,12]
   - 0.3080458641052246 Conv_1_aggregated_features[2,13]
   + 0.9457569122314453 Conv_1_aggregated_features[2,14]
   + 0.4423830211162567 Conv_1_aggregated_features[2,15] + Conv_1_t[2,4]
   = -0.2138544023036957
 Conv_1_output_constraint[2,5]: 0.2116706073284149 Conv_0_Relu_ts[2,0]
   + 0.2319677472114563 Conv_0_Relu_ts[2,1]
   + 0.1967353820800781 Conv_0_Relu_ts[2,2]
   - 0.1974936425685883 Conv_0_Relu_ts[2,3]
   + 0.087493434548378 Conv_0_Relu_ts[2,4]
   + 0.0226023308932781 Conv_0_Relu_ts[2,5]
   - 0.1455883532762527 Conv_0_Relu_ts[2,6]
   - 0.1098284423351288 Conv_0_Relu_ts[2,7]
   + 0.0375414304435253 Conv_0_Relu_ts[2,8]
   + 0.2209681868553162 Conv_0_Relu_ts[2,9]
   + 0.0180207490921021 Conv_0_Relu_ts[2,10]
   - 0.0944192707538605 Conv_0_Relu_ts[2,11]
   + 0.1318628787994385 Conv_0_Relu_ts[2,12]
   + 0.1718421876430511 Conv_0_Relu_ts[2,13]
   - 0.0992768108844757 Conv_0_Relu_ts[2,14]
   + 0.2220577001571655 Conv_0_Relu_ts[2,15]
   + 0.0500434041023254 Conv_1_aggregated_features[2,0]
   + 0.2417693138122559 Conv_1_aggregated_features[2,1]
   + 0.1884889900684357 Conv_1_aggregated_features[2,2]
   - 0.0459566116333008 Conv_1_aggregated_features[2,3]
   - 0.1095028519630432 Conv_1_aggregated_features[2,4]
   - 0.1494371891021729 Conv_1_aggregated_features[2,5]
   - 0.0601528584957123 Conv_1_aggregated_features[2,6]
   - 0.2431226968765259 Conv_1_aggregated_features[2,7]
   + 0.0709281861782074 Conv_1_aggregated_features[2,8]
   + 0.1859873235225677 Conv_1_aggregated_features[2,9]
   - 0.0025022029876709 Conv_1_aggregated_features[2,10]
   + 0.1165908575057983 Conv_1_aggregated_features[2,11]
   - 0.0946725904941559 Conv_1_aggregated_features[2,12]
   + 0.1679102778434753 Conv_1_aggregated_features[2,13]
   + 0.0888539850711823 Conv_1_aggregated_features[2,14]
   + 0.0481749773025513 Conv_1_aggregated_features[2,15] + Conv_1_t[2,5]
   = -0.2017861604690552
 Conv_1_output_constraint[2,6]: - 0.5161762237548828 Conv_0_Relu_ts[2,0]
   - 0.9039249420166016 Conv_0_Relu_ts[2,1]
   - 0.5091935992240906 Conv_0_Relu_ts[2,2]
   + 0.177293062210083 Conv_0_Relu_ts[2,3]
   + 1.534894466400146 Conv_0_Relu_ts[2,4]
   + 0.3099820613861084 Conv_0_Relu_ts[2,5]
   + 1.335353136062622 Conv_0_Relu_ts[2,6]
   - 0.3965733051300049 Conv_0_Relu_ts[2,7]
   + 0.0226722583174706 Conv_0_Relu_ts[2,8]
   - 0.0153551828116179 Conv_0_Relu_ts[2,9]
   - 0.188335582613945 Conv_0_Relu_ts[2,10]
   + 0.1465377509593964 Conv_0_Relu_ts[2,11]
   - 0.4378408491611481 Conv_0_Relu_ts[2,12]
   + 0.5950998067855835 Conv_0_Relu_ts[2,13]
   - 0.2773410975933075 Conv_0_Relu_ts[2,14]
   - 0.3855878710746765 Conv_0_Relu_ts[2,15]
   + 0.0145458607003093 Conv_1_aggregated_features[2,0]
   - 1.349490284919739 Conv_1_aggregated_features[2,1]
   - 0.6738612055778503 Conv_1_aggregated_features[2,2]
   - 0.1977581083774567 Conv_1_aggregated_features[2,3]
   + 3.616869688034058 Conv_1_aggregated_features[2,4]
   - 0.5846084952354431 Conv_1_aggregated_features[2,5]
   + 0.3911246955394745 Conv_1_aggregated_features[2,6]
   - 0.2603789269924164 Conv_1_aggregated_features[2,7]
   + 0.0530429184436798 Conv_1_aggregated_features[2,8]
   + 0.7039647698402405 Conv_1_aggregated_features[2,9]
   - 0.4948081970214844 Conv_1_aggregated_features[2,10]
   - 1.33747124671936 Conv_1_aggregated_features[2,11]
   + 0.4521820247173309 Conv_1_aggregated_features[2,12]
   + 2.879487276077271 Conv_1_aggregated_features[2,13]
   + 0.6494181156158447 Conv_1_aggregated_features[2,14]
   - 0.0441654324531555 Conv_1_aggregated_features[2,15] + Conv_1_t[2,6]
   = -1.141431212425232
 Conv_1_output_constraint[2,7]: 0.1232976838946342 Conv_0_Relu_ts[2,0]
   + 0.7059634327888489 Conv_0_Relu_ts[2,1]
   + 0.0986589044332504 Conv_0_Relu_ts[2,2]
   - 0.1159188747406006 Conv_0_Relu_ts[2,3]
   + 0.1207201555371284 Conv_0_Relu_ts[2,4]
   + 0.4950260519981384 Conv_0_Relu_ts[2,5]
   - 0.1985135078430176 Conv_0_Relu_ts[2,6]
   + 0.3846564292907715 Conv_0_Relu_ts[2,7]
   - 0.2091868817806244 Conv_0_Relu_ts[2,8]
   + 0.0282646510750055 Conv_0_Relu_ts[2,9]
   + 0.00616153934970498 Conv_0_Relu_ts[2,10]
   - 0.2992357313632965 Conv_0_Relu_ts[2,11]
   + 0.6717308759689331 Conv_0_Relu_ts[2,12]
   + 0.3884595632553101 Conv_0_Relu_ts[2,13]
   + 0.1088093742728233 Conv_0_Relu_ts[2,14]
   + 0.1363743990659714 Conv_0_Relu_ts[2,15]
   + 0.2181874513626099 Conv_1_aggregated_features[2,0]
   + 0.1884859055280685 Conv_1_aggregated_features[2,1]
   + 0.1986039727926254 Conv_1_aggregated_features[2,2]
   - 0.0569350719451904 Conv_1_aggregated_features[2,3]
   + 0.4957895576953888 Conv_1_aggregated_features[2,4]
   - 0.2215360552072525 Conv_1_aggregated_features[2,5]
   - 0.3126825392246246 Conv_1_aggregated_features[2,6]
   + 0.2129529714584351 Conv_1_aggregated_features[2,7]
   + 0.1905750632286072 Conv_1_aggregated_features[2,8]
   + 0.3835417926311493 Conv_1_aggregated_features[2,9]
   + 0.0515264272689819 Conv_1_aggregated_features[2,10]
   + 0.2774861454963684 Conv_1_aggregated_features[2,11]
   + 0.1720549166202545 Conv_1_aggregated_features[2,12]
   + 0.2042496353387833 Conv_1_aggregated_features[2,13]
   + 0.1717791706323624 Conv_1_aggregated_features[2,14]
   - 0.0635386854410172 Conv_1_aggregated_features[2,15] + Conv_1_t[2,7]
   = -0.1961010098457336
 Conv_1_output_constraint[2,8]: - 0.0984712317585945 Conv_0_Relu_ts[2,0]
   - 0.8972246050834656 Conv_0_Relu_ts[2,1]
   - 0.1342054009437561 Conv_0_Relu_ts[2,2]
   + 0.0934775173664093 Conv_0_Relu_ts[2,3]
   + 3.929172039031982 Conv_0_Relu_ts[2,4]
   - 1.699812531471252 Conv_0_Relu_ts[2,5]
   + 0.4780299663543701 Conv_0_Relu_ts[2,6]
   + 0.1916014850139618 Conv_0_Relu_ts[2,7]
   - 0.0282735247164965 Conv_0_Relu_ts[2,8]
   - 0.10334312915802 Conv_0_Relu_ts[2,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[2,10]
   - 0.9000989198684692 Conv_0_Relu_ts[2,11]
   - 0.3381493389606476 Conv_0_Relu_ts[2,12]
   + 0.2327869236469269 Conv_0_Relu_ts[2,13]
   + 0.6162272691726685 Conv_0_Relu_ts[2,14]
   - 0.1045716181397438 Conv_0_Relu_ts[2,15]
   - 0.0895406752824783 Conv_1_aggregated_features[2,0]
   - 0.0733596011996269 Conv_1_aggregated_features[2,1]
   - 0.6064443588256836 Conv_1_aggregated_features[2,2]
   + 0.1685465276241302 Conv_1_aggregated_features[2,3]
   + 2.521227836608887 Conv_1_aggregated_features[2,4]
   + 0.8868938088417053 Conv_1_aggregated_features[2,5]
   - 0.3730944097042084 Conv_1_aggregated_features[2,6]
   + 0.4433300495147705 Conv_1_aggregated_features[2,7]
   + 0.2465403079986572 Conv_1_aggregated_features[2,8]
   + 0.1199371814727783 Conv_1_aggregated_features[2,9]
   - 0.427333652973175 Conv_1_aggregated_features[2,10]
   - 1.286244511604309 Conv_1_aggregated_features[2,11]
   + 0.4271916449069977 Conv_1_aggregated_features[2,12]
   + 1.475327014923096 Conv_1_aggregated_features[2,13]
   - 0.2351342439651489 Conv_1_aggregated_features[2,14]
   - 0.1273849010467529 Conv_1_aggregated_features[2,15] + Conv_1_t[2,8]
   = -1.303357601165771
 Conv_1_output_constraint[2,9]: 0.1534582376480103 Conv_0_Relu_ts[2,0]
   + 1.456407189369202 Conv_0_Relu_ts[2,1]
   - 0.0228729248046875 Conv_0_Relu_ts[2,2]
   + 0.1227743625640869 Conv_0_Relu_ts[2,3]
   + 0.7793847918510437 Conv_0_Relu_ts[2,4]
   - 0.2757052779197693 Conv_0_Relu_ts[2,5]
   + 0.6606881022453308 Conv_0_Relu_ts[2,6]
   - 1.107266664505005 Conv_0_Relu_ts[2,7]
   + 0.1723826229572296 Conv_0_Relu_ts[2,8]
   - 0.5540673732757568 Conv_0_Relu_ts[2,9]
   + 0.3133971095085144 Conv_0_Relu_ts[2,10]
   + 0.9318023324012756 Conv_0_Relu_ts[2,11]
   + 0.5855908989906311 Conv_0_Relu_ts[2,12]
   - 0.2116718590259552 Conv_0_Relu_ts[2,13]
   - 0.3693177700042725 Conv_0_Relu_ts[2,14]
   - 0.0642165616154671 Conv_0_Relu_ts[2,15]
   + 0.1819062232971191 Conv_1_aggregated_features[2,0]
   + 1.311912417411804 Conv_1_aggregated_features[2,1]
   + 0.3460695445537567 Conv_1_aggregated_features[2,2]
   + 0.1086616516113281 Conv_1_aggregated_features[2,3]
   - 0.6099227070808411 Conv_1_aggregated_features[2,4]
   + 0.3786990940570831 Conv_1_aggregated_features[2,5]
   + 0.9473716020584106 Conv_1_aggregated_features[2,6]
   + 0.2196423262357712 Conv_1_aggregated_features[2,7]
   + 0.0830722749233246 Conv_1_aggregated_features[2,8]
   - 0.4822347462177277 Conv_1_aggregated_features[2,9]
   + 0.5108910202980042 Conv_1_aggregated_features[2,10]
   + 3.214276790618896 Conv_1_aggregated_features[2,11]
   - 0.7224436402320862 Conv_1_aggregated_features[2,12]
   + 0.2678262591362 Conv_1_aggregated_features[2,13]
   - 0.8807565569877625 Conv_1_aggregated_features[2,14]
   - 0.3268090188503265 Conv_1_aggregated_features[2,15] + Conv_1_t[2,9]
   = -0.8283404111862183
 Conv_1_output_constraint[2,10]: 0.3653882741928101 Conv_0_Relu_ts[2,0]
   - 0.8208209872245789 Conv_0_Relu_ts[2,1]
   + 0.3088222146034241 Conv_0_Relu_ts[2,2]
   + 0.1173163652420044 Conv_0_Relu_ts[2,3]
   + 0.658578634262085 Conv_0_Relu_ts[2,4]
   + 0.584019660949707 Conv_0_Relu_ts[2,5]
   - 0.8654046058654785 Conv_0_Relu_ts[2,6]
   + 0.3697350919246674 Conv_0_Relu_ts[2,7]
   - 0.2274059355258942 Conv_0_Relu_ts[2,8]
   + 1.197938442230225 Conv_0_Relu_ts[2,9]
   + 0.2446341812610626 Conv_0_Relu_ts[2,10]
   - 0.5676743984222412 Conv_0_Relu_ts[2,11]
   - 0.3123646676540375 Conv_0_Relu_ts[2,12]
   + 3.181503534317017 Conv_0_Relu_ts[2,13]
   + 0.6926240921020508 Conv_0_Relu_ts[2,14]
   + 0.0886927545070648 Conv_0_Relu_ts[2,15]
   - 0.2005636394023895 Conv_1_aggregated_features[2,0]
   + 1.403087973594666 Conv_1_aggregated_features[2,1]
   + 0.0523448958992958 Conv_1_aggregated_features[2,2]
   + 0.2411543428897858 Conv_1_aggregated_features[2,3]
   + 0.5953171253204346 Conv_1_aggregated_features[2,4]
   - 0.8278964757919312 Conv_1_aggregated_features[2,5]
   + 0.2499217540025711 Conv_1_aggregated_features[2,6]
   - 0.7393644452095032 Conv_1_aggregated_features[2,7]
   + 0.0902875959873199 Conv_1_aggregated_features[2,8]
   + 0.1488711386919022 Conv_1_aggregated_features[2,9]
   - 0.0303693022578955 Conv_1_aggregated_features[2,10]
   + 2.024244070053101 Conv_1_aggregated_features[2,11]
   + 0.495431512594223 Conv_1_aggregated_features[2,12]
   + 0.3463623225688934 Conv_1_aggregated_features[2,13]
   - 0.0763808861374855 Conv_1_aggregated_features[2,14]
   + 0.1597562879323959 Conv_1_aggregated_features[2,15] + Conv_1_t[2,10]
   = -0.5063248872756958
 Conv_1_output_constraint[2,11]: - 0.5185666680335999 Conv_0_Relu_ts[2,0]
   + 3.656918525695801 Conv_0_Relu_ts[2,1]
   - 0.7082476615905762 Conv_0_Relu_ts[2,2]
   + 0.1922211349010468 Conv_0_Relu_ts[2,3]
   - 0.7149283289909363 Conv_0_Relu_ts[2,4]
   - 3.255658626556396 Conv_0_Relu_ts[2,5]
   + 2.464392185211182 Conv_0_Relu_ts[2,6]
   - 0.9477378129959106 Conv_0_Relu_ts[2,7]
   + 0.168237030506134 Conv_0_Relu_ts[2,8]
   - 1.177640557289124 Conv_0_Relu_ts[2,9]
   - 0.7062549591064453 Conv_0_Relu_ts[2,10]
   + 1.453806400299072 Conv_0_Relu_ts[2,11]
   + 0.2612711191177368 Conv_0_Relu_ts[2,12]
   + 0.2999382615089417 Conv_0_Relu_ts[2,13]
   - 1.255579590797424 Conv_0_Relu_ts[2,14]
   - 0.4265820384025574 Conv_0_Relu_ts[2,15]
   + 0.3379200994968414 Conv_1_aggregated_features[2,0]
   - 1.33229672908783 Conv_1_aggregated_features[2,1]
   - 0.1594242304563522 Conv_1_aggregated_features[2,2]
   + 0.2298935651779175 Conv_1_aggregated_features[2,3]
   - 1.311494588851929 Conv_1_aggregated_features[2,4]
   + 3.605161666870117 Conv_1_aggregated_features[2,5]
   - 0.3176752626895905 Conv_1_aggregated_features[2,6]
   + 0.3926282525062561 Conv_1_aggregated_features[2,7]
   + 0.1479305028915405 Conv_1_aggregated_features[2,8]
   - 0.0372937023639679 Conv_1_aggregated_features[2,9]
   + 0.3455427587032318 Conv_1_aggregated_features[2,10]
   - 2.002299070358276 Conv_1_aggregated_features[2,11]
   - 0.9904713034629822 Conv_1_aggregated_features[2,12]
   + 0.0751789212226868 Conv_1_aggregated_features[2,13]
   - 0.5166233777999878 Conv_1_aggregated_features[2,14]
   + 0.0510688796639442 Conv_1_aggregated_features[2,15] + Conv_1_t[2,11]
   = -1.480785727500916
 Conv_1_output_constraint[2,12]: 0.3080185949802399 Conv_0_Relu_ts[2,0]
   + 0.0994751453399658 Conv_0_Relu_ts[2,1]
   + 0.4488444924354553 Conv_0_Relu_ts[2,2]
   + 0.0305773317813873 Conv_0_Relu_ts[2,3]
   - 1.673393130302429 Conv_0_Relu_ts[2,4]
   + 0.2824500501155853 Conv_0_Relu_ts[2,5]
   - 1.192709684371948 Conv_0_Relu_ts[2,6]
   + 1.454243063926697 Conv_0_Relu_ts[2,7]
   + 0.1514801830053329 Conv_0_Relu_ts[2,8]
   + 1.026351928710938 Conv_0_Relu_ts[2,9]
   + 0.1183762699365616 Conv_0_Relu_ts[2,10]
   + 0.352224588394165 Conv_0_Relu_ts[2,11]
   + 1.090276837348938 Conv_0_Relu_ts[2,12]
   - 2.053535461425781 Conv_0_Relu_ts[2,13]
   + 2.538641452789307 Conv_0_Relu_ts[2,14]
   + 0.6387560963630676 Conv_0_Relu_ts[2,15]
   - 0.1990496516227722 Conv_1_aggregated_features[2,0]
   + 1.247365713119507 Conv_1_aggregated_features[2,1]
   + 0.2213435918092728 Conv_1_aggregated_features[2,2]
   + 0.2398849725723267 Conv_1_aggregated_features[2,3]
   - 2.001479864120483 Conv_1_aggregated_features[2,4]
   - 1.084800243377686 Conv_1_aggregated_features[2,5]
   + 0.4287458956241608 Conv_1_aggregated_features[2,6]
   - 0.8179780840873718 Conv_1_aggregated_features[2,7]
   - 0.0883614718914032 Conv_1_aggregated_features[2,8]
   - 0.4398066997528076 Conv_1_aggregated_features[2,9]
   + 0.2760448455810547 Conv_1_aggregated_features[2,10]
   + 1.944519639015198 Conv_1_aggregated_features[2,11]
   + 1.041437149047852 Conv_1_aggregated_features[2,12]
   - 1.682050704956055 Conv_1_aggregated_features[2,13]
   - 0.1825132220983505 Conv_1_aggregated_features[2,14]
   + 0.2881160080432892 Conv_1_aggregated_features[2,15] + Conv_1_t[2,12]
   = 2.457676649093628
 Conv_1_output_constraint[2,13]: 0.2853958606719971 Conv_0_Relu_ts[2,0]
   + 0.0445350483059883 Conv_0_Relu_ts[2,1]
   - 0.0597315393388271 Conv_0_Relu_ts[2,2]
   + 0.2403226792812347 Conv_0_Relu_ts[2,3]
   + 0.5103238224983215 Conv_0_Relu_ts[2,4]
   - 0.1369927227497101 Conv_0_Relu_ts[2,5]
   + 0.5091087818145752 Conv_0_Relu_ts[2,6]
   - 0.0576926171779633 Conv_0_Relu_ts[2,7]
   - 0.00701320171356201 Conv_0_Relu_ts[2,8]
   + 0.5782948136329651 Conv_0_Relu_ts[2,9]
   - 0.1188536584377289 Conv_0_Relu_ts[2,10]
   - 0.078026682138443 Conv_0_Relu_ts[2,11]
   + 0.1965750753879547 Conv_0_Relu_ts[2,12]
   + 0.4982857704162598 Conv_0_Relu_ts[2,13]
   - 0.0995257869362831 Conv_0_Relu_ts[2,14]
   + 0.0712818428874016 Conv_0_Relu_ts[2,15]
   + 0.2628078460693359 Conv_1_aggregated_features[2,0]
   + 0.6577805280685425 Conv_1_aggregated_features[2,1]
   - 0.1415485441684723 Conv_1_aggregated_features[2,2]
   + 0.1597791016101837 Conv_1_aggregated_features[2,3]
   + 0.3969745337963104 Conv_1_aggregated_features[2,4]
   - 0.3941676616668701 Conv_1_aggregated_features[2,5]
   + 0.3484204709529877 Conv_1_aggregated_features[2,6]
   + 0.2601790428161621 Conv_1_aggregated_features[2,7]
   - 0.0652478933334351 Conv_1_aggregated_features[2,8]
   + 0.3492280542850494 Conv_1_aggregated_features[2,9]
   + 0.1917867064476013 Conv_1_aggregated_features[2,10]
   + 0.2439181953668594 Conv_1_aggregated_features[2,11]
   + 0.1009163707494736 Conv_1_aggregated_features[2,12]
   + 0.2350947707891464 Conv_1_aggregated_features[2,13]
   + 0.2627962231636047 Conv_1_aggregated_features[2,14]
   - 0.0959920063614845 Conv_1_aggregated_features[2,15] + Conv_1_t[2,13]
   = -0.4268372058868408
 Conv_1_output_constraint[2,14]: - 0.2448655962944031 Conv_0_Relu_ts[2,0]
   + 0.2688764035701752 Conv_0_Relu_ts[2,1]
   + 0.0118266940116882 Conv_0_Relu_ts[2,2]
   - 0.0670525431632996 Conv_0_Relu_ts[2,3]
   + 0.1511579602956772 Conv_0_Relu_ts[2,4]
   - 0.4796198606491089 Conv_0_Relu_ts[2,5]
   - 0.1699391901493073 Conv_0_Relu_ts[2,6]
   + 0.2647148370742798 Conv_0_Relu_ts[2,7]
   + 0.137117475271225 Conv_0_Relu_ts[2,8]
   + 0.1737731248140335 Conv_0_Relu_ts[2,9]
   - 0.1610838174819946 Conv_0_Relu_ts[2,10]
   - 0.6311220526695251 Conv_0_Relu_ts[2,11]
   + 0.6263757348060608 Conv_0_Relu_ts[2,12]
   - 0.4823384284973145 Conv_0_Relu_ts[2,13]
   + 0.2192765176296234 Conv_0_Relu_ts[2,14]
   + 0.1739131808280945 Conv_0_Relu_ts[2,15]
   + 0.0443203188478947 Conv_1_aggregated_features[2,0]
   + 0.7668815851211548 Conv_1_aggregated_features[2,1]
   + 0.248365581035614 Conv_1_aggregated_features[2,2]
   - 0.2433404624462128 Conv_1_aggregated_features[2,3]
   + 0.2770798802375793 Conv_1_aggregated_features[2,4]
   - 1.224791765213013 Conv_1_aggregated_features[2,5]
   - 0.1892879456281662 Conv_1_aggregated_features[2,6]
   + 0.1141589805483818 Conv_1_aggregated_features[2,7]
   + 0.2022489905357361 Conv_1_aggregated_features[2,8]
   - 0.0391396470367908 Conv_1_aggregated_features[2,9]
   - 0.0935317650437355 Conv_1_aggregated_features[2,10]
   - 0.5407891273498535 Conv_1_aggregated_features[2,11]
   + 0.3130630552768707 Conv_1_aggregated_features[2,12]
   + 0.3410882651805878 Conv_1_aggregated_features[2,13]
   - 1.855417490005493 Conv_1_aggregated_features[2,14]
   + 0.247613787651062 Conv_1_aggregated_features[2,15] + Conv_1_t[2,14]
   = -0.2306040376424789
 Conv_1_output_constraint[2,15]: 0.043729841709137 Conv_0_Relu_ts[2,0]
   + 0.045797374099493 Conv_0_Relu_ts[2,1]
   - 0.1762047111988068 Conv_0_Relu_ts[2,2]
   - 0.2225021123886108 Conv_0_Relu_ts[2,3]
   + 0.1344186365604401 Conv_0_Relu_ts[2,4]
   + 0.2975046932697296 Conv_0_Relu_ts[2,5]
   + 0.1135610193014145 Conv_0_Relu_ts[2,6]
   + 0.165085032582283 Conv_0_Relu_ts[2,7]
   + 0.0632660016417503 Conv_0_Relu_ts[2,8]
   + 0.2534148395061493 Conv_0_Relu_ts[2,9]
   + 0.1100017726421356 Conv_0_Relu_ts[2,10]
   + 0.045901358127594 Conv_0_Relu_ts[2,11]
   + 0.2102706432342529 Conv_0_Relu_ts[2,12]
   + 0.1767551600933075 Conv_0_Relu_ts[2,13]
   - 0.0864512398838997 Conv_0_Relu_ts[2,14]
   + 0.1616824418306351 Conv_0_Relu_ts[2,15]
   + 0.0113421082496643 Conv_1_aggregated_features[2,0]
   + 0.2387615889310837 Conv_1_aggregated_features[2,1]
   + 0.1763307154178619 Conv_1_aggregated_features[2,2]
   + 0.1442219316959381 Conv_1_aggregated_features[2,3]
   + 0.3573774099349976 Conv_1_aggregated_features[2,4]
   - 0.0406427308917046 Conv_1_aggregated_features[2,5]
   + 0.1155626475811005 Conv_1_aggregated_features[2,6]
   + 0.0186243671923876 Conv_1_aggregated_features[2,7]
   - 0.1389756500720978 Conv_1_aggregated_features[2,8]
   + 0.274962306022644 Conv_1_aggregated_features[2,9]
   + 0.0309249386191368 Conv_1_aggregated_features[2,10]
   - 0.1606657803058624 Conv_1_aggregated_features[2,11]
   + 0.2456246316432953 Conv_1_aggregated_features[2,12]
   + 0.4163841009140015 Conv_1_aggregated_features[2,13]
   + 0.29454505443573 Conv_1_aggregated_features[2,14]
   + 0.1688967496156693 Conv_1_aggregated_features[2,15] + Conv_1_t[2,15]
   = -0.112743653357029
 Conv_1_output_constraint[3,0]: - 0.3061666786670685 Conv_0_Relu_ts[3,0]
   + 0.8854666948318481 Conv_0_Relu_ts[3,1]
   - 1.108363628387451 Conv_0_Relu_ts[3,2]
   - 0.0485447347164154 Conv_0_Relu_ts[3,3]
   - 0.3054571747779846 Conv_0_Relu_ts[3,4]
   - 1.666001558303833 Conv_0_Relu_ts[3,5]
   + 1.575538277626038 Conv_0_Relu_ts[3,6]
   - 0.8786032199859619 Conv_0_Relu_ts[3,7]
   + 0.1696695387363434 Conv_0_Relu_ts[3,8]
   - 0.2058112174272537 Conv_0_Relu_ts[3,9]
   - 0.8619717359542847 Conv_0_Relu_ts[3,10]
   + 1.268710613250732 Conv_0_Relu_ts[3,11]
   + 0.1005195453763008 Conv_0_Relu_ts[3,12]
   - 0.00627355137839913 Conv_0_Relu_ts[3,13]
   - 0.6727985143661499 Conv_0_Relu_ts[3,14]
   - 0.6082888841629028 Conv_0_Relu_ts[3,15]
   + 0.0746944323182106 Conv_1_aggregated_features[3,0]
   - 0.5317568182945251 Conv_1_aggregated_features[3,1]
   + 0.4405037760734558 Conv_1_aggregated_features[3,2]
   - 0.1488203704357147 Conv_1_aggregated_features[3,3]
   + 2.01956033706665 Conv_1_aggregated_features[3,4]
   + 0.8369019031524658 Conv_1_aggregated_features[3,5]
   + 0.1561367213726044 Conv_1_aggregated_features[3,6]
   + 0.1698544919490814 Conv_1_aggregated_features[3,7]
   - 0.0660476386547089 Conv_1_aggregated_features[3,8]
   + 1.570046067237854 Conv_1_aggregated_features[3,9]
   + 0.5540977120399475 Conv_1_aggregated_features[3,10]
   - 1.275410532951355 Conv_1_aggregated_features[3,11]
   - 0.092900924384594 Conv_1_aggregated_features[3,12]
   + 3.287018537521362 Conv_1_aggregated_features[3,13]
   + 2.392175912857056 Conv_1_aggregated_features[3,14]
   + 0.3933185040950775 Conv_1_aggregated_features[3,15] + Conv_1_t[3,0]
   = -2.192610263824463
 Conv_1_output_constraint[3,1]: 0.5202509164810181 Conv_0_Relu_ts[3,0]
   + 0.6686695218086243 Conv_0_Relu_ts[3,1]
   - 0.4720649719238281 Conv_0_Relu_ts[3,2]
   + 0.1026173532009125 Conv_0_Relu_ts[3,3]
   + 0.2593795359134674 Conv_0_Relu_ts[3,4]
   - 2.458898544311523 Conv_0_Relu_ts[3,5]
   - 0.5632196664810181 Conv_0_Relu_ts[3,6]
   - 1.791601657867432 Conv_0_Relu_ts[3,7]
   - 0.1268838942050934 Conv_0_Relu_ts[3,8]
   + 0.7120329737663269 Conv_0_Relu_ts[3,9]
   - 0.5212016701698303 Conv_0_Relu_ts[3,10]
   + 0.6136131286621094 Conv_0_Relu_ts[3,11]
   + 1.563590168952942 Conv_0_Relu_ts[3,12]
   - 0.3512922525405884 Conv_0_Relu_ts[3,13]
   + 0.5421575903892517 Conv_0_Relu_ts[3,14]
   + 0.7331249117851257 Conv_0_Relu_ts[3,15]
   + 0.0113700022920966 Conv_1_aggregated_features[3,0]
   + 2.169116258621216 Conv_1_aggregated_features[3,1]
   + 0.00623909616842866 Conv_1_aggregated_features[3,2]
   + 0.0937856733798981 Conv_1_aggregated_features[3,3]
   + 0.8328568339347839 Conv_1_aggregated_features[3,4]
   + 0.383074939250946 Conv_1_aggregated_features[3,5]
   - 0.1267539262771606 Conv_1_aggregated_features[3,6]
   + 0.6484018564224243 Conv_1_aggregated_features[3,7]
   + 0.1870926916599274 Conv_1_aggregated_features[3,8]
   - 0.4116691052913666 Conv_1_aggregated_features[3,9]
   + 0.1439376026391983 Conv_1_aggregated_features[3,10]
   + 1.05384635925293 Conv_1_aggregated_features[3,11]
   + 1.155439853668213 Conv_1_aggregated_features[3,12]
   - 0.2060132175683975 Conv_1_aggregated_features[3,13]
   + 0.6003169417381287 Conv_1_aggregated_features[3,14]
   + 0.1458597332239151 Conv_1_aggregated_features[3,15] + Conv_1_t[3,1]
   = -0.1077260375022888
 Conv_1_output_constraint[3,2]: 0.4269110858440399 Conv_0_Relu_ts[3,0]
   - 1.097789287567139 Conv_0_Relu_ts[3,1]
   - 0.3380239307880402 Conv_0_Relu_ts[3,2]
   + 0.23756343126297 Conv_0_Relu_ts[3,3]
   - 0.337738424539566 Conv_0_Relu_ts[3,4]
   - 1.407688736915588 Conv_0_Relu_ts[3,5]
   - 0.7854894995689392 Conv_0_Relu_ts[3,6]
   - 1.080899715423584 Conv_0_Relu_ts[3,7]
   + 0.0233876463025808 Conv_0_Relu_ts[3,8]
   + 1.849825739860535 Conv_0_Relu_ts[3,9]
   + 0.4580322206020355 Conv_0_Relu_ts[3,10]
   - 0.819361686706543 Conv_0_Relu_ts[3,11]
   - 0.0822571739554405 Conv_0_Relu_ts[3,12]
   - 0.1501271426677704 Conv_0_Relu_ts[3,13]
   + 0.1831801384687424 Conv_0_Relu_ts[3,14]
   + 0.00308510870672762 Conv_0_Relu_ts[3,15]
   - 0.2153502702713013 Conv_1_aggregated_features[3,0]
   + 3.817557334899902 Conv_1_aggregated_features[3,1]
   - 0.3402683734893799 Conv_1_aggregated_features[3,2]
   + 0.0650523602962494 Conv_1_aggregated_features[3,3]
   - 0.4435569643974304 Conv_1_aggregated_features[3,4]
   + 0.045248981565237 Conv_1_aggregated_features[3,5]
   + 0.844337522983551 Conv_1_aggregated_features[3,6]
   - 0.0691490024328232 Conv_1_aggregated_features[3,7]
   + 0.0830505788326263 Conv_1_aggregated_features[3,8]
   - 0.3022302687168121 Conv_1_aggregated_features[3,9]
   - 0.611566960811615 Conv_1_aggregated_features[3,10]
   + 3.582128524780273 Conv_1_aggregated_features[3,11]
   + 0.1314867585897446 Conv_1_aggregated_features[3,12]
   + 0.5333523154258728 Conv_1_aggregated_features[3,13]
   - 0.1983102858066559 Conv_1_aggregated_features[3,14]
   + 0.1628565639257431 Conv_1_aggregated_features[3,15] + Conv_1_t[3,2]
   = -0.4443503022193909
 Conv_1_output_constraint[3,3]: 0.066770151257515 Conv_0_Relu_ts[3,0]
   + 0.3136637210845947 Conv_0_Relu_ts[3,1]
   + 0.2078424692153931 Conv_0_Relu_ts[3,2]
   - 0.1673517823219299 Conv_0_Relu_ts[3,3]
   + 1.022224068641663 Conv_0_Relu_ts[3,4]
   - 0.1723377108573914 Conv_0_Relu_ts[3,5]
   + 0.3032898306846619 Conv_0_Relu_ts[3,6]
   + 0.0841449126601219 Conv_0_Relu_ts[3,7]
   + 0.0921086072921753 Conv_0_Relu_ts[3,8]
   - 0.022469462826848 Conv_0_Relu_ts[3,9]
   - 0.1657154262065887 Conv_0_Relu_ts[3,10]
   + 1.007031679153442 Conv_0_Relu_ts[3,11]
   + 0.0208180602639914 Conv_0_Relu_ts[3,12]
   + 0.3933880031108856 Conv_0_Relu_ts[3,13]
   + 0.3077191412448883 Conv_0_Relu_ts[3,14]
   + 0.0400979965925217 Conv_0_Relu_ts[3,15]
   + 0.0230858232825994 Conv_1_aggregated_features[3,0]
   + 0.3792195022106171 Conv_1_aggregated_features[3,1]
   + 0.1572305560112 Conv_1_aggregated_features[3,2]
   - 0.0165807008743286 Conv_1_aggregated_features[3,3]
   + 0.5649795532226562 Conv_1_aggregated_features[3,4]
   + 0.6253551840782166 Conv_1_aggregated_features[3,5]
   + 0.3968201577663422 Conv_1_aggregated_features[3,6]
   + 0.2728661298751831 Conv_1_aggregated_features[3,7]
   - 0.210608571767807 Conv_1_aggregated_features[3,8]
   + 0.094737209379673 Conv_1_aggregated_features[3,9]
   - 0.1354089975357056 Conv_1_aggregated_features[3,10]
   + 0.9330839514732361 Conv_1_aggregated_features[3,11]
   + 0.1677136421203613 Conv_1_aggregated_features[3,12]
   + 0.6573846936225891 Conv_1_aggregated_features[3,13]
   + 0.044663343578577 Conv_1_aggregated_features[3,14]
   - 0.0132347755134106 Conv_1_aggregated_features[3,15] + Conv_1_t[3,3]
   = -0.327608048915863
 Conv_1_output_constraint[3,4]: - 0.0720538049936295 Conv_0_Relu_ts[3,0]
   + 3.730369091033936 Conv_0_Relu_ts[3,1]
   - 0.8118388652801514 Conv_0_Relu_ts[3,2]
   - 0.174347460269928 Conv_0_Relu_ts[3,3]
   - 3.084089279174805 Conv_0_Relu_ts[3,4]
   - 5.146248340606689 Conv_0_Relu_ts[3,5]
   - 2.440467357635498 Conv_0_Relu_ts[3,6]
   - 3.765322923660278 Conv_0_Relu_ts[3,7]
   + 0.153628796339035 Conv_0_Relu_ts[3,8]
   + 0.1013353765010834 Conv_0_Relu_ts[3,9]
   - 0.5803871154785156 Conv_0_Relu_ts[3,10]
   - 2.657181739807129 Conv_0_Relu_ts[3,11]
   + 0.6840037703514099 Conv_0_Relu_ts[3,12]
   - 1.184461355209351 Conv_0_Relu_ts[3,13]
   - 0.3772841989994049 Conv_0_Relu_ts[3,14]
   + 0.5856359601020813 Conv_0_Relu_ts[3,15]
   + 0.2062403708696365 Conv_1_aggregated_features[3,0]
   + 1.470675468444824 Conv_1_aggregated_features[3,1]
   + 0.3523006737232208 Conv_1_aggregated_features[3,2]
   - 0.1351108849048615 Conv_1_aggregated_features[3,3]
   + 0.8932182788848877 Conv_1_aggregated_features[3,4]
   + 1.194340229034424 Conv_1_aggregated_features[3,5]
   + 0.0163087919354439 Conv_1_aggregated_features[3,6]
   - 0.5082771182060242 Conv_1_aggregated_features[3,7]
   - 0.0201514065265656 Conv_1_aggregated_features[3,8]
   + 0.9045064449310303 Conv_1_aggregated_features[3,9]
   + 0.7243757843971252 Conv_1_aggregated_features[3,10]
   + 0.5823414921760559 Conv_1_aggregated_features[3,11]
   + 0.0738087296485901 Conv_1_aggregated_features[3,12]
   - 0.3080458641052246 Conv_1_aggregated_features[3,13]
   + 0.9457569122314453 Conv_1_aggregated_features[3,14]
   + 0.4423830211162567 Conv_1_aggregated_features[3,15] + Conv_1_t[3,4]
   = -0.2138544023036957
 Conv_1_output_constraint[3,5]: 0.2116706073284149 Conv_0_Relu_ts[3,0]
   + 0.2319677472114563 Conv_0_Relu_ts[3,1]
   + 0.1967353820800781 Conv_0_Relu_ts[3,2]
   - 0.1974936425685883 Conv_0_Relu_ts[3,3]
   + 0.087493434548378 Conv_0_Relu_ts[3,4]
   + 0.0226023308932781 Conv_0_Relu_ts[3,5]
   - 0.1455883532762527 Conv_0_Relu_ts[3,6]
   - 0.1098284423351288 Conv_0_Relu_ts[3,7]
   + 0.0375414304435253 Conv_0_Relu_ts[3,8]
   + 0.2209681868553162 Conv_0_Relu_ts[3,9]
   + 0.0180207490921021 Conv_0_Relu_ts[3,10]
   - 0.0944192707538605 Conv_0_Relu_ts[3,11]
   + 0.1318628787994385 Conv_0_Relu_ts[3,12]
   + 0.1718421876430511 Conv_0_Relu_ts[3,13]
   - 0.0992768108844757 Conv_0_Relu_ts[3,14]
   + 0.2220577001571655 Conv_0_Relu_ts[3,15]
   + 0.0500434041023254 Conv_1_aggregated_features[3,0]
   + 0.2417693138122559 Conv_1_aggregated_features[3,1]
   + 0.1884889900684357 Conv_1_aggregated_features[3,2]
   - 0.0459566116333008 Conv_1_aggregated_features[3,3]
   - 0.1095028519630432 Conv_1_aggregated_features[3,4]
   - 0.1494371891021729 Conv_1_aggregated_features[3,5]
   - 0.0601528584957123 Conv_1_aggregated_features[3,6]
   - 0.2431226968765259 Conv_1_aggregated_features[3,7]
   + 0.0709281861782074 Conv_1_aggregated_features[3,8]
   + 0.1859873235225677 Conv_1_aggregated_features[3,9]
   - 0.0025022029876709 Conv_1_aggregated_features[3,10]
   + 0.1165908575057983 Conv_1_aggregated_features[3,11]
   - 0.0946725904941559 Conv_1_aggregated_features[3,12]
   + 0.1679102778434753 Conv_1_aggregated_features[3,13]
   + 0.0888539850711823 Conv_1_aggregated_features[3,14]
   + 0.0481749773025513 Conv_1_aggregated_features[3,15] + Conv_1_t[3,5]
   = -0.2017861604690552
 Conv_1_output_constraint[3,6]: - 0.5161762237548828 Conv_0_Relu_ts[3,0]
   - 0.9039249420166016 Conv_0_Relu_ts[3,1]
   - 0.5091935992240906 Conv_0_Relu_ts[3,2]
   + 0.177293062210083 Conv_0_Relu_ts[3,3]
   + 1.534894466400146 Conv_0_Relu_ts[3,4]
   + 0.3099820613861084 Conv_0_Relu_ts[3,5]
   + 1.335353136062622 Conv_0_Relu_ts[3,6]
   - 0.3965733051300049 Conv_0_Relu_ts[3,7]
   + 0.0226722583174706 Conv_0_Relu_ts[3,8]
   - 0.0153551828116179 Conv_0_Relu_ts[3,9]
   - 0.188335582613945 Conv_0_Relu_ts[3,10]
   + 0.1465377509593964 Conv_0_Relu_ts[3,11]
   - 0.4378408491611481 Conv_0_Relu_ts[3,12]
   + 0.5950998067855835 Conv_0_Relu_ts[3,13]
   - 0.2773410975933075 Conv_0_Relu_ts[3,14]
   - 0.3855878710746765 Conv_0_Relu_ts[3,15]
   + 0.0145458607003093 Conv_1_aggregated_features[3,0]
   - 1.349490284919739 Conv_1_aggregated_features[3,1]
   - 0.6738612055778503 Conv_1_aggregated_features[3,2]
   - 0.1977581083774567 Conv_1_aggregated_features[3,3]
   + 3.616869688034058 Conv_1_aggregated_features[3,4]
   - 0.5846084952354431 Conv_1_aggregated_features[3,5]
   + 0.3911246955394745 Conv_1_aggregated_features[3,6]
   - 0.2603789269924164 Conv_1_aggregated_features[3,7]
   + 0.0530429184436798 Conv_1_aggregated_features[3,8]
   + 0.7039647698402405 Conv_1_aggregated_features[3,9]
   - 0.4948081970214844 Conv_1_aggregated_features[3,10]
   - 1.33747124671936 Conv_1_aggregated_features[3,11]
   + 0.4521820247173309 Conv_1_aggregated_features[3,12]
   + 2.879487276077271 Conv_1_aggregated_features[3,13]
   + 0.6494181156158447 Conv_1_aggregated_features[3,14]
   - 0.0441654324531555 Conv_1_aggregated_features[3,15] + Conv_1_t[3,6]
   = -1.141431212425232
 Conv_1_output_constraint[3,7]: 0.1232976838946342 Conv_0_Relu_ts[3,0]
   + 0.7059634327888489 Conv_0_Relu_ts[3,1]
   + 0.0986589044332504 Conv_0_Relu_ts[3,2]
   - 0.1159188747406006 Conv_0_Relu_ts[3,3]
   + 0.1207201555371284 Conv_0_Relu_ts[3,4]
   + 0.4950260519981384 Conv_0_Relu_ts[3,5]
   - 0.1985135078430176 Conv_0_Relu_ts[3,6]
   + 0.3846564292907715 Conv_0_Relu_ts[3,7]
   - 0.2091868817806244 Conv_0_Relu_ts[3,8]
   + 0.0282646510750055 Conv_0_Relu_ts[3,9]
   + 0.00616153934970498 Conv_0_Relu_ts[3,10]
   - 0.2992357313632965 Conv_0_Relu_ts[3,11]
   + 0.6717308759689331 Conv_0_Relu_ts[3,12]
   + 0.3884595632553101 Conv_0_Relu_ts[3,13]
   + 0.1088093742728233 Conv_0_Relu_ts[3,14]
   + 0.1363743990659714 Conv_0_Relu_ts[3,15]
   + 0.2181874513626099 Conv_1_aggregated_features[3,0]
   + 0.1884859055280685 Conv_1_aggregated_features[3,1]
   + 0.1986039727926254 Conv_1_aggregated_features[3,2]
   - 0.0569350719451904 Conv_1_aggregated_features[3,3]
   + 0.4957895576953888 Conv_1_aggregated_features[3,4]
   - 0.2215360552072525 Conv_1_aggregated_features[3,5]
   - 0.3126825392246246 Conv_1_aggregated_features[3,6]
   + 0.2129529714584351 Conv_1_aggregated_features[3,7]
   + 0.1905750632286072 Conv_1_aggregated_features[3,8]
   + 0.3835417926311493 Conv_1_aggregated_features[3,9]
   + 0.0515264272689819 Conv_1_aggregated_features[3,10]
   + 0.2774861454963684 Conv_1_aggregated_features[3,11]
   + 0.1720549166202545 Conv_1_aggregated_features[3,12]
   + 0.2042496353387833 Conv_1_aggregated_features[3,13]
   + 0.1717791706323624 Conv_1_aggregated_features[3,14]
   - 0.0635386854410172 Conv_1_aggregated_features[3,15] + Conv_1_t[3,7]
   = -0.1961010098457336
 Conv_1_output_constraint[3,8]: - 0.0984712317585945 Conv_0_Relu_ts[3,0]
   - 0.8972246050834656 Conv_0_Relu_ts[3,1]
   - 0.1342054009437561 Conv_0_Relu_ts[3,2]
   + 0.0934775173664093 Conv_0_Relu_ts[3,3]
   + 3.929172039031982 Conv_0_Relu_ts[3,4]
   - 1.699812531471252 Conv_0_Relu_ts[3,5]
   + 0.4780299663543701 Conv_0_Relu_ts[3,6]
   + 0.1916014850139618 Conv_0_Relu_ts[3,7]
   - 0.0282735247164965 Conv_0_Relu_ts[3,8]
   - 0.10334312915802 Conv_0_Relu_ts[3,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[3,10]
   - 0.9000989198684692 Conv_0_Relu_ts[3,11]
   - 0.3381493389606476 Conv_0_Relu_ts[3,12]
   + 0.2327869236469269 Conv_0_Relu_ts[3,13]
   + 0.6162272691726685 Conv_0_Relu_ts[3,14]
   - 0.1045716181397438 Conv_0_Relu_ts[3,15]
   - 0.0895406752824783 Conv_1_aggregated_features[3,0]
   - 0.0733596011996269 Conv_1_aggregated_features[3,1]
   - 0.6064443588256836 Conv_1_aggregated_features[3,2]
   + 0.1685465276241302 Conv_1_aggregated_features[3,3]
   + 2.521227836608887 Conv_1_aggregated_features[3,4]
   + 0.8868938088417053 Conv_1_aggregated_features[3,5]
   - 0.3730944097042084 Conv_1_aggregated_features[3,6]
   + 0.4433300495147705 Conv_1_aggregated_features[3,7]
   + 0.2465403079986572 Conv_1_aggregated_features[3,8]
   + 0.1199371814727783 Conv_1_aggregated_features[3,9]
   - 0.427333652973175 Conv_1_aggregated_features[3,10]
   - 1.286244511604309 Conv_1_aggregated_features[3,11]
   + 0.4271916449069977 Conv_1_aggregated_features[3,12]
   + 1.475327014923096 Conv_1_aggregated_features[3,13]
   - 0.2351342439651489 Conv_1_aggregated_features[3,14]
   - 0.1273849010467529 Conv_1_aggregated_features[3,15] + Conv_1_t[3,8]
   = -1.303357601165771
 Conv_1_output_constraint[3,9]: 0.1534582376480103 Conv_0_Relu_ts[3,0]
   + 1.456407189369202 Conv_0_Relu_ts[3,1]
   - 0.0228729248046875 Conv_0_Relu_ts[3,2]
   + 0.1227743625640869 Conv_0_Relu_ts[3,3]
   + 0.7793847918510437 Conv_0_Relu_ts[3,4]
   - 0.2757052779197693 Conv_0_Relu_ts[3,5]
   + 0.6606881022453308 Conv_0_Relu_ts[3,6]
   - 1.107266664505005 Conv_0_Relu_ts[3,7]
   + 0.1723826229572296 Conv_0_Relu_ts[3,8]
   - 0.5540673732757568 Conv_0_Relu_ts[3,9]
   + 0.3133971095085144 Conv_0_Relu_ts[3,10]
   + 0.9318023324012756 Conv_0_Relu_ts[3,11]
   + 0.5855908989906311 Conv_0_Relu_ts[3,12]
   - 0.2116718590259552 Conv_0_Relu_ts[3,13]
   - 0.3693177700042725 Conv_0_Relu_ts[3,14]
   - 0.0642165616154671 Conv_0_Relu_ts[3,15]
   + 0.1819062232971191 Conv_1_aggregated_features[3,0]
   + 1.311912417411804 Conv_1_aggregated_features[3,1]
   + 0.3460695445537567 Conv_1_aggregated_features[3,2]
   + 0.1086616516113281 Conv_1_aggregated_features[3,3]
   - 0.6099227070808411 Conv_1_aggregated_features[3,4]
   + 0.3786990940570831 Conv_1_aggregated_features[3,5]
   + 0.9473716020584106 Conv_1_aggregated_features[3,6]
   + 0.2196423262357712 Conv_1_aggregated_features[3,7]
   + 0.0830722749233246 Conv_1_aggregated_features[3,8]
   - 0.4822347462177277 Conv_1_aggregated_features[3,9]
   + 0.5108910202980042 Conv_1_aggregated_features[3,10]
   + 3.214276790618896 Conv_1_aggregated_features[3,11]
   - 0.7224436402320862 Conv_1_aggregated_features[3,12]
   + 0.2678262591362 Conv_1_aggregated_features[3,13]
   - 0.8807565569877625 Conv_1_aggregated_features[3,14]
   - 0.3268090188503265 Conv_1_aggregated_features[3,15] + Conv_1_t[3,9]
   = -0.8283404111862183
 Conv_1_output_constraint[3,10]: 0.3653882741928101 Conv_0_Relu_ts[3,0]
   - 0.8208209872245789 Conv_0_Relu_ts[3,1]
   + 0.3088222146034241 Conv_0_Relu_ts[3,2]
   + 0.1173163652420044 Conv_0_Relu_ts[3,3]
   + 0.658578634262085 Conv_0_Relu_ts[3,4]
   + 0.584019660949707 Conv_0_Relu_ts[3,5]
   - 0.8654046058654785 Conv_0_Relu_ts[3,6]
   + 0.3697350919246674 Conv_0_Relu_ts[3,7]
   - 0.2274059355258942 Conv_0_Relu_ts[3,8]
   + 1.197938442230225 Conv_0_Relu_ts[3,9]
   + 0.2446341812610626 Conv_0_Relu_ts[3,10]
   - 0.5676743984222412 Conv_0_Relu_ts[3,11]
   - 0.3123646676540375 Conv_0_Relu_ts[3,12]
   + 3.181503534317017 Conv_0_Relu_ts[3,13]
   + 0.6926240921020508 Conv_0_Relu_ts[3,14]
   + 0.0886927545070648 Conv_0_Relu_ts[3,15]
   - 0.2005636394023895 Conv_1_aggregated_features[3,0]
   + 1.403087973594666 Conv_1_aggregated_features[3,1]
   + 0.0523448958992958 Conv_1_aggregated_features[3,2]
   + 0.2411543428897858 Conv_1_aggregated_features[3,3]
   + 0.5953171253204346 Conv_1_aggregated_features[3,4]
   - 0.8278964757919312 Conv_1_aggregated_features[3,5]
   + 0.2499217540025711 Conv_1_aggregated_features[3,6]
   - 0.7393644452095032 Conv_1_aggregated_features[3,7]
   + 0.0902875959873199 Conv_1_aggregated_features[3,8]
   + 0.1488711386919022 Conv_1_aggregated_features[3,9]
   - 0.0303693022578955 Conv_1_aggregated_features[3,10]
   + 2.024244070053101 Conv_1_aggregated_features[3,11]
   + 0.495431512594223 Conv_1_aggregated_features[3,12]
   + 0.3463623225688934 Conv_1_aggregated_features[3,13]
   - 0.0763808861374855 Conv_1_aggregated_features[3,14]
   + 0.1597562879323959 Conv_1_aggregated_features[3,15] + Conv_1_t[3,10]
   = -0.5063248872756958
 Conv_1_output_constraint[3,11]: - 0.5185666680335999 Conv_0_Relu_ts[3,0]
   + 3.656918525695801 Conv_0_Relu_ts[3,1]
   - 0.7082476615905762 Conv_0_Relu_ts[3,2]
   + 0.1922211349010468 Conv_0_Relu_ts[3,3]
   - 0.7149283289909363 Conv_0_Relu_ts[3,4]
   - 3.255658626556396 Conv_0_Relu_ts[3,5]
   + 2.464392185211182 Conv_0_Relu_ts[3,6]
   - 0.9477378129959106 Conv_0_Relu_ts[3,7]
   + 0.168237030506134 Conv_0_Relu_ts[3,8]
   - 1.177640557289124 Conv_0_Relu_ts[3,9]
   - 0.7062549591064453 Conv_0_Relu_ts[3,10]
   + 1.453806400299072 Conv_0_Relu_ts[3,11]
   + 0.2612711191177368 Conv_0_Relu_ts[3,12]
   + 0.2999382615089417 Conv_0_Relu_ts[3,13]
   - 1.255579590797424 Conv_0_Relu_ts[3,14]
   - 0.4265820384025574 Conv_0_Relu_ts[3,15]
   + 0.3379200994968414 Conv_1_aggregated_features[3,0]
   - 1.33229672908783 Conv_1_aggregated_features[3,1]
   - 0.1594242304563522 Conv_1_aggregated_features[3,2]
   + 0.2298935651779175 Conv_1_aggregated_features[3,3]
   - 1.311494588851929 Conv_1_aggregated_features[3,4]
   + 3.605161666870117 Conv_1_aggregated_features[3,5]
   - 0.3176752626895905 Conv_1_aggregated_features[3,6]
   + 0.3926282525062561 Conv_1_aggregated_features[3,7]
   + 0.1479305028915405 Conv_1_aggregated_features[3,8]
   - 0.0372937023639679 Conv_1_aggregated_features[3,9]
   + 0.3455427587032318 Conv_1_aggregated_features[3,10]
   - 2.002299070358276 Conv_1_aggregated_features[3,11]
   - 0.9904713034629822 Conv_1_aggregated_features[3,12]
   + 0.0751789212226868 Conv_1_aggregated_features[3,13]
   - 0.5166233777999878 Conv_1_aggregated_features[3,14]
   + 0.0510688796639442 Conv_1_aggregated_features[3,15] + Conv_1_t[3,11]
   = -1.480785727500916
 Conv_1_output_constraint[3,12]: 0.3080185949802399 Conv_0_Relu_ts[3,0]
   + 0.0994751453399658 Conv_0_Relu_ts[3,1]
   + 0.4488444924354553 Conv_0_Relu_ts[3,2]
   + 0.0305773317813873 Conv_0_Relu_ts[3,3]
   - 1.673393130302429 Conv_0_Relu_ts[3,4]
   + 0.2824500501155853 Conv_0_Relu_ts[3,5]
   - 1.192709684371948 Conv_0_Relu_ts[3,6]
   + 1.454243063926697 Conv_0_Relu_ts[3,7]
   + 0.1514801830053329 Conv_0_Relu_ts[3,8]
   + 1.026351928710938 Conv_0_Relu_ts[3,9]
   + 0.1183762699365616 Conv_0_Relu_ts[3,10]
   + 0.352224588394165 Conv_0_Relu_ts[3,11]
   + 1.090276837348938 Conv_0_Relu_ts[3,12]
   - 2.053535461425781 Conv_0_Relu_ts[3,13]
   + 2.538641452789307 Conv_0_Relu_ts[3,14]
   + 0.6387560963630676 Conv_0_Relu_ts[3,15]
   - 0.1990496516227722 Conv_1_aggregated_features[3,0]
   + 1.247365713119507 Conv_1_aggregated_features[3,1]
   + 0.2213435918092728 Conv_1_aggregated_features[3,2]
   + 0.2398849725723267 Conv_1_aggregated_features[3,3]
   - 2.001479864120483 Conv_1_aggregated_features[3,4]
   - 1.084800243377686 Conv_1_aggregated_features[3,5]
   + 0.4287458956241608 Conv_1_aggregated_features[3,6]
   - 0.8179780840873718 Conv_1_aggregated_features[3,7]
   - 0.0883614718914032 Conv_1_aggregated_features[3,8]
   - 0.4398066997528076 Conv_1_aggregated_features[3,9]
   + 0.2760448455810547 Conv_1_aggregated_features[3,10]
   + 1.944519639015198 Conv_1_aggregated_features[3,11]
   + 1.041437149047852 Conv_1_aggregated_features[3,12]
   - 1.682050704956055 Conv_1_aggregated_features[3,13]
   - 0.1825132220983505 Conv_1_aggregated_features[3,14]
   + 0.2881160080432892 Conv_1_aggregated_features[3,15] + Conv_1_t[3,12]
   = 2.457676649093628
 Conv_1_output_constraint[3,13]: 0.2853958606719971 Conv_0_Relu_ts[3,0]
   + 0.0445350483059883 Conv_0_Relu_ts[3,1]
   - 0.0597315393388271 Conv_0_Relu_ts[3,2]
   + 0.2403226792812347 Conv_0_Relu_ts[3,3]
   + 0.5103238224983215 Conv_0_Relu_ts[3,4]
   - 0.1369927227497101 Conv_0_Relu_ts[3,5]
   + 0.5091087818145752 Conv_0_Relu_ts[3,6]
   - 0.0576926171779633 Conv_0_Relu_ts[3,7]
   - 0.00701320171356201 Conv_0_Relu_ts[3,8]
   + 0.5782948136329651 Conv_0_Relu_ts[3,9]
   - 0.1188536584377289 Conv_0_Relu_ts[3,10]
   - 0.078026682138443 Conv_0_Relu_ts[3,11]
   + 0.1965750753879547 Conv_0_Relu_ts[3,12]
   + 0.4982857704162598 Conv_0_Relu_ts[3,13]
   - 0.0995257869362831 Conv_0_Relu_ts[3,14]
   + 0.0712818428874016 Conv_0_Relu_ts[3,15]
   + 0.2628078460693359 Conv_1_aggregated_features[3,0]
   + 0.6577805280685425 Conv_1_aggregated_features[3,1]
   - 0.1415485441684723 Conv_1_aggregated_features[3,2]
   + 0.1597791016101837 Conv_1_aggregated_features[3,3]
   + 0.3969745337963104 Conv_1_aggregated_features[3,4]
   - 0.3941676616668701 Conv_1_aggregated_features[3,5]
   + 0.3484204709529877 Conv_1_aggregated_features[3,6]
   + 0.2601790428161621 Conv_1_aggregated_features[3,7]
   - 0.0652478933334351 Conv_1_aggregated_features[3,8]
   + 0.3492280542850494 Conv_1_aggregated_features[3,9]
   + 0.1917867064476013 Conv_1_aggregated_features[3,10]
   + 0.2439181953668594 Conv_1_aggregated_features[3,11]
   + 0.1009163707494736 Conv_1_aggregated_features[3,12]
   + 0.2350947707891464 Conv_1_aggregated_features[3,13]
   + 0.2627962231636047 Conv_1_aggregated_features[3,14]
   - 0.0959920063614845 Conv_1_aggregated_features[3,15] + Conv_1_t[3,13]
   = -0.4268372058868408
 Conv_1_output_constraint[3,14]: - 0.2448655962944031 Conv_0_Relu_ts[3,0]
   + 0.2688764035701752 Conv_0_Relu_ts[3,1]
   + 0.0118266940116882 Conv_0_Relu_ts[3,2]
   - 0.0670525431632996 Conv_0_Relu_ts[3,3]
   + 0.1511579602956772 Conv_0_Relu_ts[3,4]
   - 0.4796198606491089 Conv_0_Relu_ts[3,5]
   - 0.1699391901493073 Conv_0_Relu_ts[3,6]
   + 0.2647148370742798 Conv_0_Relu_ts[3,7]
   + 0.137117475271225 Conv_0_Relu_ts[3,8]
   + 0.1737731248140335 Conv_0_Relu_ts[3,9]
   - 0.1610838174819946 Conv_0_Relu_ts[3,10]
   - 0.6311220526695251 Conv_0_Relu_ts[3,11]
   + 0.6263757348060608 Conv_0_Relu_ts[3,12]
   - 0.4823384284973145 Conv_0_Relu_ts[3,13]
   + 0.2192765176296234 Conv_0_Relu_ts[3,14]
   + 0.1739131808280945 Conv_0_Relu_ts[3,15]
   + 0.0443203188478947 Conv_1_aggregated_features[3,0]
   + 0.7668815851211548 Conv_1_aggregated_features[3,1]
   + 0.248365581035614 Conv_1_aggregated_features[3,2]
   - 0.2433404624462128 Conv_1_aggregated_features[3,3]
   + 0.2770798802375793 Conv_1_aggregated_features[3,4]
   - 1.224791765213013 Conv_1_aggregated_features[3,5]
   - 0.1892879456281662 Conv_1_aggregated_features[3,6]
   + 0.1141589805483818 Conv_1_aggregated_features[3,7]
   + 0.2022489905357361 Conv_1_aggregated_features[3,8]
   - 0.0391396470367908 Conv_1_aggregated_features[3,9]
   - 0.0935317650437355 Conv_1_aggregated_features[3,10]
   - 0.5407891273498535 Conv_1_aggregated_features[3,11]
   + 0.3130630552768707 Conv_1_aggregated_features[3,12]
   + 0.3410882651805878 Conv_1_aggregated_features[3,13]
   - 1.855417490005493 Conv_1_aggregated_features[3,14]
   + 0.247613787651062 Conv_1_aggregated_features[3,15] + Conv_1_t[3,14]
   = -0.2306040376424789
 Conv_1_output_constraint[3,15]: 0.043729841709137 Conv_0_Relu_ts[3,0]
   + 0.045797374099493 Conv_0_Relu_ts[3,1]
   - 0.1762047111988068 Conv_0_Relu_ts[3,2]
   - 0.2225021123886108 Conv_0_Relu_ts[3,3]
   + 0.1344186365604401 Conv_0_Relu_ts[3,4]
   + 0.2975046932697296 Conv_0_Relu_ts[3,5]
   + 0.1135610193014145 Conv_0_Relu_ts[3,6]
   + 0.165085032582283 Conv_0_Relu_ts[3,7]
   + 0.0632660016417503 Conv_0_Relu_ts[3,8]
   + 0.2534148395061493 Conv_0_Relu_ts[3,9]
   + 0.1100017726421356 Conv_0_Relu_ts[3,10]
   + 0.045901358127594 Conv_0_Relu_ts[3,11]
   + 0.2102706432342529 Conv_0_Relu_ts[3,12]
   + 0.1767551600933075 Conv_0_Relu_ts[3,13]
   - 0.0864512398838997 Conv_0_Relu_ts[3,14]
   + 0.1616824418306351 Conv_0_Relu_ts[3,15]
   + 0.0113421082496643 Conv_1_aggregated_features[3,0]
   + 0.2387615889310837 Conv_1_aggregated_features[3,1]
   + 0.1763307154178619 Conv_1_aggregated_features[3,2]
   + 0.1442219316959381 Conv_1_aggregated_features[3,3]
   + 0.3573774099349976 Conv_1_aggregated_features[3,4]
   - 0.0406427308917046 Conv_1_aggregated_features[3,5]
   + 0.1155626475811005 Conv_1_aggregated_features[3,6]
   + 0.0186243671923876 Conv_1_aggregated_features[3,7]
   - 0.1389756500720978 Conv_1_aggregated_features[3,8]
   + 0.274962306022644 Conv_1_aggregated_features[3,9]
   + 0.0309249386191368 Conv_1_aggregated_features[3,10]
   - 0.1606657803058624 Conv_1_aggregated_features[3,11]
   + 0.2456246316432953 Conv_1_aggregated_features[3,12]
   + 0.4163841009140015 Conv_1_aggregated_features[3,13]
   + 0.29454505443573 Conv_1_aggregated_features[3,14]
   + 0.1688967496156693 Conv_1_aggregated_features[3,15] + Conv_1_t[3,15]
   = -0.112743653357029
 Conv_1_output_constraint[4,0]: - 0.3061666786670685 Conv_0_Relu_ts[4,0]
   + 0.8854666948318481 Conv_0_Relu_ts[4,1]
   - 1.108363628387451 Conv_0_Relu_ts[4,2]
   - 0.0485447347164154 Conv_0_Relu_ts[4,3]
   - 0.3054571747779846 Conv_0_Relu_ts[4,4]
   - 1.666001558303833 Conv_0_Relu_ts[4,5]
   + 1.575538277626038 Conv_0_Relu_ts[4,6]
   - 0.8786032199859619 Conv_0_Relu_ts[4,7]
   + 0.1696695387363434 Conv_0_Relu_ts[4,8]
   - 0.2058112174272537 Conv_0_Relu_ts[4,9]
   - 0.8619717359542847 Conv_0_Relu_ts[4,10]
   + 1.268710613250732 Conv_0_Relu_ts[4,11]
   + 0.1005195453763008 Conv_0_Relu_ts[4,12]
   - 0.00627355137839913 Conv_0_Relu_ts[4,13]
   - 0.6727985143661499 Conv_0_Relu_ts[4,14]
   - 0.6082888841629028 Conv_0_Relu_ts[4,15]
   + 0.0746944323182106 Conv_1_aggregated_features[4,0]
   - 0.5317568182945251 Conv_1_aggregated_features[4,1]
   + 0.4405037760734558 Conv_1_aggregated_features[4,2]
   - 0.1488203704357147 Conv_1_aggregated_features[4,3]
   + 2.01956033706665 Conv_1_aggregated_features[4,4]
   + 0.8369019031524658 Conv_1_aggregated_features[4,5]
   + 0.1561367213726044 Conv_1_aggregated_features[4,6]
   + 0.1698544919490814 Conv_1_aggregated_features[4,7]
   - 0.0660476386547089 Conv_1_aggregated_features[4,8]
   + 1.570046067237854 Conv_1_aggregated_features[4,9]
   + 0.5540977120399475 Conv_1_aggregated_features[4,10]
   - 1.275410532951355 Conv_1_aggregated_features[4,11]
   - 0.092900924384594 Conv_1_aggregated_features[4,12]
   + 3.287018537521362 Conv_1_aggregated_features[4,13]
   + 2.392175912857056 Conv_1_aggregated_features[4,14]
   + 0.3933185040950775 Conv_1_aggregated_features[4,15] + Conv_1_t[4,0]
   = -2.192610263824463
 Conv_1_output_constraint[4,1]: 0.5202509164810181 Conv_0_Relu_ts[4,0]
   + 0.6686695218086243 Conv_0_Relu_ts[4,1]
   - 0.4720649719238281 Conv_0_Relu_ts[4,2]
   + 0.1026173532009125 Conv_0_Relu_ts[4,3]
   + 0.2593795359134674 Conv_0_Relu_ts[4,4]
   - 2.458898544311523 Conv_0_Relu_ts[4,5]
   - 0.5632196664810181 Conv_0_Relu_ts[4,6]
   - 1.791601657867432 Conv_0_Relu_ts[4,7]
   - 0.1268838942050934 Conv_0_Relu_ts[4,8]
   + 0.7120329737663269 Conv_0_Relu_ts[4,9]
   - 0.5212016701698303 Conv_0_Relu_ts[4,10]
   + 0.6136131286621094 Conv_0_Relu_ts[4,11]
   + 1.563590168952942 Conv_0_Relu_ts[4,12]
   - 0.3512922525405884 Conv_0_Relu_ts[4,13]
   + 0.5421575903892517 Conv_0_Relu_ts[4,14]
   + 0.7331249117851257 Conv_0_Relu_ts[4,15]
   + 0.0113700022920966 Conv_1_aggregated_features[4,0]
   + 2.169116258621216 Conv_1_aggregated_features[4,1]
   + 0.00623909616842866 Conv_1_aggregated_features[4,2]
   + 0.0937856733798981 Conv_1_aggregated_features[4,3]
   + 0.8328568339347839 Conv_1_aggregated_features[4,4]
   + 0.383074939250946 Conv_1_aggregated_features[4,5]
   - 0.1267539262771606 Conv_1_aggregated_features[4,6]
   + 0.6484018564224243 Conv_1_aggregated_features[4,7]
   + 0.1870926916599274 Conv_1_aggregated_features[4,8]
   - 0.4116691052913666 Conv_1_aggregated_features[4,9]
   + 0.1439376026391983 Conv_1_aggregated_features[4,10]
   + 1.05384635925293 Conv_1_aggregated_features[4,11]
   + 1.155439853668213 Conv_1_aggregated_features[4,12]
   - 0.2060132175683975 Conv_1_aggregated_features[4,13]
   + 0.6003169417381287 Conv_1_aggregated_features[4,14]
   + 0.1458597332239151 Conv_1_aggregated_features[4,15] + Conv_1_t[4,1]
   = -0.1077260375022888
 Conv_1_output_constraint[4,2]: 0.4269110858440399 Conv_0_Relu_ts[4,0]
   - 1.097789287567139 Conv_0_Relu_ts[4,1]
   - 0.3380239307880402 Conv_0_Relu_ts[4,2]
   + 0.23756343126297 Conv_0_Relu_ts[4,3]
   - 0.337738424539566 Conv_0_Relu_ts[4,4]
   - 1.407688736915588 Conv_0_Relu_ts[4,5]
   - 0.7854894995689392 Conv_0_Relu_ts[4,6]
   - 1.080899715423584 Conv_0_Relu_ts[4,7]
   + 0.0233876463025808 Conv_0_Relu_ts[4,8]
   + 1.849825739860535 Conv_0_Relu_ts[4,9]
   + 0.4580322206020355 Conv_0_Relu_ts[4,10]
   - 0.819361686706543 Conv_0_Relu_ts[4,11]
   - 0.0822571739554405 Conv_0_Relu_ts[4,12]
   - 0.1501271426677704 Conv_0_Relu_ts[4,13]
   + 0.1831801384687424 Conv_0_Relu_ts[4,14]
   + 0.00308510870672762 Conv_0_Relu_ts[4,15]
   - 0.2153502702713013 Conv_1_aggregated_features[4,0]
   + 3.817557334899902 Conv_1_aggregated_features[4,1]
   - 0.3402683734893799 Conv_1_aggregated_features[4,2]
   + 0.0650523602962494 Conv_1_aggregated_features[4,3]
   - 0.4435569643974304 Conv_1_aggregated_features[4,4]
   + 0.045248981565237 Conv_1_aggregated_features[4,5]
   + 0.844337522983551 Conv_1_aggregated_features[4,6]
   - 0.0691490024328232 Conv_1_aggregated_features[4,7]
   + 0.0830505788326263 Conv_1_aggregated_features[4,8]
   - 0.3022302687168121 Conv_1_aggregated_features[4,9]
   - 0.611566960811615 Conv_1_aggregated_features[4,10]
   + 3.582128524780273 Conv_1_aggregated_features[4,11]
   + 0.1314867585897446 Conv_1_aggregated_features[4,12]
   + 0.5333523154258728 Conv_1_aggregated_features[4,13]
   - 0.1983102858066559 Conv_1_aggregated_features[4,14]
   + 0.1628565639257431 Conv_1_aggregated_features[4,15] + Conv_1_t[4,2]
   = -0.4443503022193909
 Conv_1_output_constraint[4,3]: 0.066770151257515 Conv_0_Relu_ts[4,0]
   + 0.3136637210845947 Conv_0_Relu_ts[4,1]
   + 0.2078424692153931 Conv_0_Relu_ts[4,2]
   - 0.1673517823219299 Conv_0_Relu_ts[4,3]
   + 1.022224068641663 Conv_0_Relu_ts[4,4]
   - 0.1723377108573914 Conv_0_Relu_ts[4,5]
   + 0.3032898306846619 Conv_0_Relu_ts[4,6]
   + 0.0841449126601219 Conv_0_Relu_ts[4,7]
   + 0.0921086072921753 Conv_0_Relu_ts[4,8]
   - 0.022469462826848 Conv_0_Relu_ts[4,9]
   - 0.1657154262065887 Conv_0_Relu_ts[4,10]
   + 1.007031679153442 Conv_0_Relu_ts[4,11]
   + 0.0208180602639914 Conv_0_Relu_ts[4,12]
   + 0.3933880031108856 Conv_0_Relu_ts[4,13]
   + 0.3077191412448883 Conv_0_Relu_ts[4,14]
   + 0.0400979965925217 Conv_0_Relu_ts[4,15]
   + 0.0230858232825994 Conv_1_aggregated_features[4,0]
   + 0.3792195022106171 Conv_1_aggregated_features[4,1]
   + 0.1572305560112 Conv_1_aggregated_features[4,2]
   - 0.0165807008743286 Conv_1_aggregated_features[4,3]
   + 0.5649795532226562 Conv_1_aggregated_features[4,4]
   + 0.6253551840782166 Conv_1_aggregated_features[4,5]
   + 0.3968201577663422 Conv_1_aggregated_features[4,6]
   + 0.2728661298751831 Conv_1_aggregated_features[4,7]
   - 0.210608571767807 Conv_1_aggregated_features[4,8]
   + 0.094737209379673 Conv_1_aggregated_features[4,9]
   - 0.1354089975357056 Conv_1_aggregated_features[4,10]
   + 0.9330839514732361 Conv_1_aggregated_features[4,11]
   + 0.1677136421203613 Conv_1_aggregated_features[4,12]
   + 0.6573846936225891 Conv_1_aggregated_features[4,13]
   + 0.044663343578577 Conv_1_aggregated_features[4,14]
   - 0.0132347755134106 Conv_1_aggregated_features[4,15] + Conv_1_t[4,3]
   = -0.327608048915863
 Conv_1_output_constraint[4,4]: - 0.0720538049936295 Conv_0_Relu_ts[4,0]
   + 3.730369091033936 Conv_0_Relu_ts[4,1]
   - 0.8118388652801514 Conv_0_Relu_ts[4,2]
   - 0.174347460269928 Conv_0_Relu_ts[4,3]
   - 3.084089279174805 Conv_0_Relu_ts[4,4]
   - 5.146248340606689 Conv_0_Relu_ts[4,5]
   - 2.440467357635498 Conv_0_Relu_ts[4,6]
   - 3.765322923660278 Conv_0_Relu_ts[4,7]
   + 0.153628796339035 Conv_0_Relu_ts[4,8]
   + 0.1013353765010834 Conv_0_Relu_ts[4,9]
   - 0.5803871154785156 Conv_0_Relu_ts[4,10]
   - 2.657181739807129 Conv_0_Relu_ts[4,11]
   + 0.6840037703514099 Conv_0_Relu_ts[4,12]
   - 1.184461355209351 Conv_0_Relu_ts[4,13]
   - 0.3772841989994049 Conv_0_Relu_ts[4,14]
   + 0.5856359601020813 Conv_0_Relu_ts[4,15]
   + 0.2062403708696365 Conv_1_aggregated_features[4,0]
   + 1.470675468444824 Conv_1_aggregated_features[4,1]
   + 0.3523006737232208 Conv_1_aggregated_features[4,2]
   - 0.1351108849048615 Conv_1_aggregated_features[4,3]
   + 0.8932182788848877 Conv_1_aggregated_features[4,4]
   + 1.194340229034424 Conv_1_aggregated_features[4,5]
   + 0.0163087919354439 Conv_1_aggregated_features[4,6]
   - 0.5082771182060242 Conv_1_aggregated_features[4,7]
   - 0.0201514065265656 Conv_1_aggregated_features[4,8]
   + 0.9045064449310303 Conv_1_aggregated_features[4,9]
   + 0.7243757843971252 Conv_1_aggregated_features[4,10]
   + 0.5823414921760559 Conv_1_aggregated_features[4,11]
   + 0.0738087296485901 Conv_1_aggregated_features[4,12]
   - 0.3080458641052246 Conv_1_aggregated_features[4,13]
   + 0.9457569122314453 Conv_1_aggregated_features[4,14]
   + 0.4423830211162567 Conv_1_aggregated_features[4,15] + Conv_1_t[4,4]
   = -0.2138544023036957
 Conv_1_output_constraint[4,5]: 0.2116706073284149 Conv_0_Relu_ts[4,0]
   + 0.2319677472114563 Conv_0_Relu_ts[4,1]
   + 0.1967353820800781 Conv_0_Relu_ts[4,2]
   - 0.1974936425685883 Conv_0_Relu_ts[4,3]
   + 0.087493434548378 Conv_0_Relu_ts[4,4]
   + 0.0226023308932781 Conv_0_Relu_ts[4,5]
   - 0.1455883532762527 Conv_0_Relu_ts[4,6]
   - 0.1098284423351288 Conv_0_Relu_ts[4,7]
   + 0.0375414304435253 Conv_0_Relu_ts[4,8]
   + 0.2209681868553162 Conv_0_Relu_ts[4,9]
   + 0.0180207490921021 Conv_0_Relu_ts[4,10]
   - 0.0944192707538605 Conv_0_Relu_ts[4,11]
   + 0.1318628787994385 Conv_0_Relu_ts[4,12]
   + 0.1718421876430511 Conv_0_Relu_ts[4,13]
   - 0.0992768108844757 Conv_0_Relu_ts[4,14]
   + 0.2220577001571655 Conv_0_Relu_ts[4,15]
   + 0.0500434041023254 Conv_1_aggregated_features[4,0]
   + 0.2417693138122559 Conv_1_aggregated_features[4,1]
   + 0.1884889900684357 Conv_1_aggregated_features[4,2]
   - 0.0459566116333008 Conv_1_aggregated_features[4,3]
   - 0.1095028519630432 Conv_1_aggregated_features[4,4]
   - 0.1494371891021729 Conv_1_aggregated_features[4,5]
   - 0.0601528584957123 Conv_1_aggregated_features[4,6]
   - 0.2431226968765259 Conv_1_aggregated_features[4,7]
   + 0.0709281861782074 Conv_1_aggregated_features[4,8]
   + 0.1859873235225677 Conv_1_aggregated_features[4,9]
   - 0.0025022029876709 Conv_1_aggregated_features[4,10]
   + 0.1165908575057983 Conv_1_aggregated_features[4,11]
   - 0.0946725904941559 Conv_1_aggregated_features[4,12]
   + 0.1679102778434753 Conv_1_aggregated_features[4,13]
   + 0.0888539850711823 Conv_1_aggregated_features[4,14]
   + 0.0481749773025513 Conv_1_aggregated_features[4,15] + Conv_1_t[4,5]
   = -0.2017861604690552
 Conv_1_output_constraint[4,6]: - 0.5161762237548828 Conv_0_Relu_ts[4,0]
   - 0.9039249420166016 Conv_0_Relu_ts[4,1]
   - 0.5091935992240906 Conv_0_Relu_ts[4,2]
   + 0.177293062210083 Conv_0_Relu_ts[4,3]
   + 1.534894466400146 Conv_0_Relu_ts[4,4]
   + 0.3099820613861084 Conv_0_Relu_ts[4,5]
   + 1.335353136062622 Conv_0_Relu_ts[4,6]
   - 0.3965733051300049 Conv_0_Relu_ts[4,7]
   + 0.0226722583174706 Conv_0_Relu_ts[4,8]
   - 0.0153551828116179 Conv_0_Relu_ts[4,9]
   - 0.188335582613945 Conv_0_Relu_ts[4,10]
   + 0.1465377509593964 Conv_0_Relu_ts[4,11]
   - 0.4378408491611481 Conv_0_Relu_ts[4,12]
   + 0.5950998067855835 Conv_0_Relu_ts[4,13]
   - 0.2773410975933075 Conv_0_Relu_ts[4,14]
   - 0.3855878710746765 Conv_0_Relu_ts[4,15]
   + 0.0145458607003093 Conv_1_aggregated_features[4,0]
   - 1.349490284919739 Conv_1_aggregated_features[4,1]
   - 0.6738612055778503 Conv_1_aggregated_features[4,2]
   - 0.1977581083774567 Conv_1_aggregated_features[4,3]
   + 3.616869688034058 Conv_1_aggregated_features[4,4]
   - 0.5846084952354431 Conv_1_aggregated_features[4,5]
   + 0.3911246955394745 Conv_1_aggregated_features[4,6]
   - 0.2603789269924164 Conv_1_aggregated_features[4,7]
   + 0.0530429184436798 Conv_1_aggregated_features[4,8]
   + 0.7039647698402405 Conv_1_aggregated_features[4,9]
   - 0.4948081970214844 Conv_1_aggregated_features[4,10]
   - 1.33747124671936 Conv_1_aggregated_features[4,11]
   + 0.4521820247173309 Conv_1_aggregated_features[4,12]
   + 2.879487276077271 Conv_1_aggregated_features[4,13]
   + 0.6494181156158447 Conv_1_aggregated_features[4,14]
   - 0.0441654324531555 Conv_1_aggregated_features[4,15] + Conv_1_t[4,6]
   = -1.141431212425232
 Conv_1_output_constraint[4,7]: 0.1232976838946342 Conv_0_Relu_ts[4,0]
   + 0.7059634327888489 Conv_0_Relu_ts[4,1]
   + 0.0986589044332504 Conv_0_Relu_ts[4,2]
   - 0.1159188747406006 Conv_0_Relu_ts[4,3]
   + 0.1207201555371284 Conv_0_Relu_ts[4,4]
   + 0.4950260519981384 Conv_0_Relu_ts[4,5]
   - 0.1985135078430176 Conv_0_Relu_ts[4,6]
   + 0.3846564292907715 Conv_0_Relu_ts[4,7]
   - 0.2091868817806244 Conv_0_Relu_ts[4,8]
   + 0.0282646510750055 Conv_0_Relu_ts[4,9]
   + 0.00616153934970498 Conv_0_Relu_ts[4,10]
   - 0.2992357313632965 Conv_0_Relu_ts[4,11]
   + 0.6717308759689331 Conv_0_Relu_ts[4,12]
   + 0.3884595632553101 Conv_0_Relu_ts[4,13]
   + 0.1088093742728233 Conv_0_Relu_ts[4,14]
   + 0.1363743990659714 Conv_0_Relu_ts[4,15]
   + 0.2181874513626099 Conv_1_aggregated_features[4,0]
   + 0.1884859055280685 Conv_1_aggregated_features[4,1]
   + 0.1986039727926254 Conv_1_aggregated_features[4,2]
   - 0.0569350719451904 Conv_1_aggregated_features[4,3]
   + 0.4957895576953888 Conv_1_aggregated_features[4,4]
   - 0.2215360552072525 Conv_1_aggregated_features[4,5]
   - 0.3126825392246246 Conv_1_aggregated_features[4,6]
   + 0.2129529714584351 Conv_1_aggregated_features[4,7]
   + 0.1905750632286072 Conv_1_aggregated_features[4,8]
   + 0.3835417926311493 Conv_1_aggregated_features[4,9]
   + 0.0515264272689819 Conv_1_aggregated_features[4,10]
   + 0.2774861454963684 Conv_1_aggregated_features[4,11]
   + 0.1720549166202545 Conv_1_aggregated_features[4,12]
   + 0.2042496353387833 Conv_1_aggregated_features[4,13]
   + 0.1717791706323624 Conv_1_aggregated_features[4,14]
   - 0.0635386854410172 Conv_1_aggregated_features[4,15] + Conv_1_t[4,7]
   = -0.1961010098457336
 Conv_1_output_constraint[4,8]: - 0.0984712317585945 Conv_0_Relu_ts[4,0]
   - 0.8972246050834656 Conv_0_Relu_ts[4,1]
   - 0.1342054009437561 Conv_0_Relu_ts[4,2]
   + 0.0934775173664093 Conv_0_Relu_ts[4,3]
   + 3.929172039031982 Conv_0_Relu_ts[4,4]
   - 1.699812531471252 Conv_0_Relu_ts[4,5]
   + 0.4780299663543701 Conv_0_Relu_ts[4,6]
   + 0.1916014850139618 Conv_0_Relu_ts[4,7]
   - 0.0282735247164965 Conv_0_Relu_ts[4,8]
   - 0.10334312915802 Conv_0_Relu_ts[4,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[4,10]
   - 0.9000989198684692 Conv_0_Relu_ts[4,11]
   - 0.3381493389606476 Conv_0_Relu_ts[4,12]
   + 0.2327869236469269 Conv_0_Relu_ts[4,13]
   + 0.6162272691726685 Conv_0_Relu_ts[4,14]
   - 0.1045716181397438 Conv_0_Relu_ts[4,15]
   - 0.0895406752824783 Conv_1_aggregated_features[4,0]
   - 0.0733596011996269 Conv_1_aggregated_features[4,1]
   - 0.6064443588256836 Conv_1_aggregated_features[4,2]
   + 0.1685465276241302 Conv_1_aggregated_features[4,3]
   + 2.521227836608887 Conv_1_aggregated_features[4,4]
   + 0.8868938088417053 Conv_1_aggregated_features[4,5]
   - 0.3730944097042084 Conv_1_aggregated_features[4,6]
   + 0.4433300495147705 Conv_1_aggregated_features[4,7]
   + 0.2465403079986572 Conv_1_aggregated_features[4,8]
   + 0.1199371814727783 Conv_1_aggregated_features[4,9]
   - 0.427333652973175 Conv_1_aggregated_features[4,10]
   - 1.286244511604309 Conv_1_aggregated_features[4,11]
   + 0.4271916449069977 Conv_1_aggregated_features[4,12]
   + 1.475327014923096 Conv_1_aggregated_features[4,13]
   - 0.2351342439651489 Conv_1_aggregated_features[4,14]
   - 0.1273849010467529 Conv_1_aggregated_features[4,15] + Conv_1_t[4,8]
   = -1.303357601165771
 Conv_1_output_constraint[4,9]: 0.1534582376480103 Conv_0_Relu_ts[4,0]
   + 1.456407189369202 Conv_0_Relu_ts[4,1]
   - 0.0228729248046875 Conv_0_Relu_ts[4,2]
   + 0.1227743625640869 Conv_0_Relu_ts[4,3]
   + 0.7793847918510437 Conv_0_Relu_ts[4,4]
   - 0.2757052779197693 Conv_0_Relu_ts[4,5]
   + 0.6606881022453308 Conv_0_Relu_ts[4,6]
   - 1.107266664505005 Conv_0_Relu_ts[4,7]
   + 0.1723826229572296 Conv_0_Relu_ts[4,8]
   - 0.5540673732757568 Conv_0_Relu_ts[4,9]
   + 0.3133971095085144 Conv_0_Relu_ts[4,10]
   + 0.9318023324012756 Conv_0_Relu_ts[4,11]
   + 0.5855908989906311 Conv_0_Relu_ts[4,12]
   - 0.2116718590259552 Conv_0_Relu_ts[4,13]
   - 0.3693177700042725 Conv_0_Relu_ts[4,14]
   - 0.0642165616154671 Conv_0_Relu_ts[4,15]
   + 0.1819062232971191 Conv_1_aggregated_features[4,0]
   + 1.311912417411804 Conv_1_aggregated_features[4,1]
   + 0.3460695445537567 Conv_1_aggregated_features[4,2]
   + 0.1086616516113281 Conv_1_aggregated_features[4,3]
   - 0.6099227070808411 Conv_1_aggregated_features[4,4]
   + 0.3786990940570831 Conv_1_aggregated_features[4,5]
   + 0.9473716020584106 Conv_1_aggregated_features[4,6]
   + 0.2196423262357712 Conv_1_aggregated_features[4,7]
   + 0.0830722749233246 Conv_1_aggregated_features[4,8]
   - 0.4822347462177277 Conv_1_aggregated_features[4,9]
   + 0.5108910202980042 Conv_1_aggregated_features[4,10]
   + 3.214276790618896 Conv_1_aggregated_features[4,11]
   - 0.7224436402320862 Conv_1_aggregated_features[4,12]
   + 0.2678262591362 Conv_1_aggregated_features[4,13]
   - 0.8807565569877625 Conv_1_aggregated_features[4,14]
   - 0.3268090188503265 Conv_1_aggregated_features[4,15] + Conv_1_t[4,9]
   = -0.8283404111862183
 Conv_1_output_constraint[4,10]: 0.3653882741928101 Conv_0_Relu_ts[4,0]
   - 0.8208209872245789 Conv_0_Relu_ts[4,1]
   + 0.3088222146034241 Conv_0_Relu_ts[4,2]
   + 0.1173163652420044 Conv_0_Relu_ts[4,3]
   + 0.658578634262085 Conv_0_Relu_ts[4,4]
   + 0.584019660949707 Conv_0_Relu_ts[4,5]
   - 0.8654046058654785 Conv_0_Relu_ts[4,6]
   + 0.3697350919246674 Conv_0_Relu_ts[4,7]
   - 0.2274059355258942 Conv_0_Relu_ts[4,8]
   + 1.197938442230225 Conv_0_Relu_ts[4,9]
   + 0.2446341812610626 Conv_0_Relu_ts[4,10]
   - 0.5676743984222412 Conv_0_Relu_ts[4,11]
   - 0.3123646676540375 Conv_0_Relu_ts[4,12]
   + 3.181503534317017 Conv_0_Relu_ts[4,13]
   + 0.6926240921020508 Conv_0_Relu_ts[4,14]
   + 0.0886927545070648 Conv_0_Relu_ts[4,15]
   - 0.2005636394023895 Conv_1_aggregated_features[4,0]
   + 1.403087973594666 Conv_1_aggregated_features[4,1]
   + 0.0523448958992958 Conv_1_aggregated_features[4,2]
   + 0.2411543428897858 Conv_1_aggregated_features[4,3]
   + 0.5953171253204346 Conv_1_aggregated_features[4,4]
   - 0.8278964757919312 Conv_1_aggregated_features[4,5]
   + 0.2499217540025711 Conv_1_aggregated_features[4,6]
   - 0.7393644452095032 Conv_1_aggregated_features[4,7]
   + 0.0902875959873199 Conv_1_aggregated_features[4,8]
   + 0.1488711386919022 Conv_1_aggregated_features[4,9]
   - 0.0303693022578955 Conv_1_aggregated_features[4,10]
   + 2.024244070053101 Conv_1_aggregated_features[4,11]
   + 0.495431512594223 Conv_1_aggregated_features[4,12]
   + 0.3463623225688934 Conv_1_aggregated_features[4,13]
   - 0.0763808861374855 Conv_1_aggregated_features[4,14]
   + 0.1597562879323959 Conv_1_aggregated_features[4,15] + Conv_1_t[4,10]
   = -0.5063248872756958
 Conv_1_output_constraint[4,11]: - 0.5185666680335999 Conv_0_Relu_ts[4,0]
   + 3.656918525695801 Conv_0_Relu_ts[4,1]
   - 0.7082476615905762 Conv_0_Relu_ts[4,2]
   + 0.1922211349010468 Conv_0_Relu_ts[4,3]
   - 0.7149283289909363 Conv_0_Relu_ts[4,4]
   - 3.255658626556396 Conv_0_Relu_ts[4,5]
   + 2.464392185211182 Conv_0_Relu_ts[4,6]
   - 0.9477378129959106 Conv_0_Relu_ts[4,7]
   + 0.168237030506134 Conv_0_Relu_ts[4,8]
   - 1.177640557289124 Conv_0_Relu_ts[4,9]
   - 0.7062549591064453 Conv_0_Relu_ts[4,10]
   + 1.453806400299072 Conv_0_Relu_ts[4,11]
   + 0.2612711191177368 Conv_0_Relu_ts[4,12]
   + 0.2999382615089417 Conv_0_Relu_ts[4,13]
   - 1.255579590797424 Conv_0_Relu_ts[4,14]
   - 0.4265820384025574 Conv_0_Relu_ts[4,15]
   + 0.3379200994968414 Conv_1_aggregated_features[4,0]
   - 1.33229672908783 Conv_1_aggregated_features[4,1]
   - 0.1594242304563522 Conv_1_aggregated_features[4,2]
   + 0.2298935651779175 Conv_1_aggregated_features[4,3]
   - 1.311494588851929 Conv_1_aggregated_features[4,4]
   + 3.605161666870117 Conv_1_aggregated_features[4,5]
   - 0.3176752626895905 Conv_1_aggregated_features[4,6]
   + 0.3926282525062561 Conv_1_aggregated_features[4,7]
   + 0.1479305028915405 Conv_1_aggregated_features[4,8]
   - 0.0372937023639679 Conv_1_aggregated_features[4,9]
   + 0.3455427587032318 Conv_1_aggregated_features[4,10]
   - 2.002299070358276 Conv_1_aggregated_features[4,11]
   - 0.9904713034629822 Conv_1_aggregated_features[4,12]
   + 0.0751789212226868 Conv_1_aggregated_features[4,13]
   - 0.5166233777999878 Conv_1_aggregated_features[4,14]
   + 0.0510688796639442 Conv_1_aggregated_features[4,15] + Conv_1_t[4,11]
   = -1.480785727500916
 Conv_1_output_constraint[4,12]: 0.3080185949802399 Conv_0_Relu_ts[4,0]
   + 0.0994751453399658 Conv_0_Relu_ts[4,1]
   + 0.4488444924354553 Conv_0_Relu_ts[4,2]
   + 0.0305773317813873 Conv_0_Relu_ts[4,3]
   - 1.673393130302429 Conv_0_Relu_ts[4,4]
   + 0.2824500501155853 Conv_0_Relu_ts[4,5]
   - 1.192709684371948 Conv_0_Relu_ts[4,6]
   + 1.454243063926697 Conv_0_Relu_ts[4,7]
   + 0.1514801830053329 Conv_0_Relu_ts[4,8]
   + 1.026351928710938 Conv_0_Relu_ts[4,9]
   + 0.1183762699365616 Conv_0_Relu_ts[4,10]
   + 0.352224588394165 Conv_0_Relu_ts[4,11]
   + 1.090276837348938 Conv_0_Relu_ts[4,12]
   - 2.053535461425781 Conv_0_Relu_ts[4,13]
   + 2.538641452789307 Conv_0_Relu_ts[4,14]
   + 0.6387560963630676 Conv_0_Relu_ts[4,15]
   - 0.1990496516227722 Conv_1_aggregated_features[4,0]
   + 1.247365713119507 Conv_1_aggregated_features[4,1]
   + 0.2213435918092728 Conv_1_aggregated_features[4,2]
   + 0.2398849725723267 Conv_1_aggregated_features[4,3]
   - 2.001479864120483 Conv_1_aggregated_features[4,4]
   - 1.084800243377686 Conv_1_aggregated_features[4,5]
   + 0.4287458956241608 Conv_1_aggregated_features[4,6]
   - 0.8179780840873718 Conv_1_aggregated_features[4,7]
   - 0.0883614718914032 Conv_1_aggregated_features[4,8]
   - 0.4398066997528076 Conv_1_aggregated_features[4,9]
   + 0.2760448455810547 Conv_1_aggregated_features[4,10]
   + 1.944519639015198 Conv_1_aggregated_features[4,11]
   + 1.041437149047852 Conv_1_aggregated_features[4,12]
   - 1.682050704956055 Conv_1_aggregated_features[4,13]
   - 0.1825132220983505 Conv_1_aggregated_features[4,14]
   + 0.2881160080432892 Conv_1_aggregated_features[4,15] + Conv_1_t[4,12]
   = 2.457676649093628
 Conv_1_output_constraint[4,13]: 0.2853958606719971 Conv_0_Relu_ts[4,0]
   + 0.0445350483059883 Conv_0_Relu_ts[4,1]
   - 0.0597315393388271 Conv_0_Relu_ts[4,2]
   + 0.2403226792812347 Conv_0_Relu_ts[4,3]
   + 0.5103238224983215 Conv_0_Relu_ts[4,4]
   - 0.1369927227497101 Conv_0_Relu_ts[4,5]
   + 0.5091087818145752 Conv_0_Relu_ts[4,6]
   - 0.0576926171779633 Conv_0_Relu_ts[4,7]
   - 0.00701320171356201 Conv_0_Relu_ts[4,8]
   + 0.5782948136329651 Conv_0_Relu_ts[4,9]
   - 0.1188536584377289 Conv_0_Relu_ts[4,10]
   - 0.078026682138443 Conv_0_Relu_ts[4,11]
   + 0.1965750753879547 Conv_0_Relu_ts[4,12]
   + 0.4982857704162598 Conv_0_Relu_ts[4,13]
   - 0.0995257869362831 Conv_0_Relu_ts[4,14]
   + 0.0712818428874016 Conv_0_Relu_ts[4,15]
   + 0.2628078460693359 Conv_1_aggregated_features[4,0]
   + 0.6577805280685425 Conv_1_aggregated_features[4,1]
   - 0.1415485441684723 Conv_1_aggregated_features[4,2]
   + 0.1597791016101837 Conv_1_aggregated_features[4,3]
   + 0.3969745337963104 Conv_1_aggregated_features[4,4]
   - 0.3941676616668701 Conv_1_aggregated_features[4,5]
   + 0.3484204709529877 Conv_1_aggregated_features[4,6]
   + 0.2601790428161621 Conv_1_aggregated_features[4,7]
   - 0.0652478933334351 Conv_1_aggregated_features[4,8]
   + 0.3492280542850494 Conv_1_aggregated_features[4,9]
   + 0.1917867064476013 Conv_1_aggregated_features[4,10]
   + 0.2439181953668594 Conv_1_aggregated_features[4,11]
   + 0.1009163707494736 Conv_1_aggregated_features[4,12]
   + 0.2350947707891464 Conv_1_aggregated_features[4,13]
   + 0.2627962231636047 Conv_1_aggregated_features[4,14]
   - 0.0959920063614845 Conv_1_aggregated_features[4,15] + Conv_1_t[4,13]
   = -0.4268372058868408
 Conv_1_output_constraint[4,14]: - 0.2448655962944031 Conv_0_Relu_ts[4,0]
   + 0.2688764035701752 Conv_0_Relu_ts[4,1]
   + 0.0118266940116882 Conv_0_Relu_ts[4,2]
   - 0.0670525431632996 Conv_0_Relu_ts[4,3]
   + 0.1511579602956772 Conv_0_Relu_ts[4,4]
   - 0.4796198606491089 Conv_0_Relu_ts[4,5]
   - 0.1699391901493073 Conv_0_Relu_ts[4,6]
   + 0.2647148370742798 Conv_0_Relu_ts[4,7]
   + 0.137117475271225 Conv_0_Relu_ts[4,8]
   + 0.1737731248140335 Conv_0_Relu_ts[4,9]
   - 0.1610838174819946 Conv_0_Relu_ts[4,10]
   - 0.6311220526695251 Conv_0_Relu_ts[4,11]
   + 0.6263757348060608 Conv_0_Relu_ts[4,12]
   - 0.4823384284973145 Conv_0_Relu_ts[4,13]
   + 0.2192765176296234 Conv_0_Relu_ts[4,14]
   + 0.1739131808280945 Conv_0_Relu_ts[4,15]
   + 0.0443203188478947 Conv_1_aggregated_features[4,0]
   + 0.7668815851211548 Conv_1_aggregated_features[4,1]
   + 0.248365581035614 Conv_1_aggregated_features[4,2]
   - 0.2433404624462128 Conv_1_aggregated_features[4,3]
   + 0.2770798802375793 Conv_1_aggregated_features[4,4]
   - 1.224791765213013 Conv_1_aggregated_features[4,5]
   - 0.1892879456281662 Conv_1_aggregated_features[4,6]
   + 0.1141589805483818 Conv_1_aggregated_features[4,7]
   + 0.2022489905357361 Conv_1_aggregated_features[4,8]
   - 0.0391396470367908 Conv_1_aggregated_features[4,9]
   - 0.0935317650437355 Conv_1_aggregated_features[4,10]
   - 0.5407891273498535 Conv_1_aggregated_features[4,11]
   + 0.3130630552768707 Conv_1_aggregated_features[4,12]
   + 0.3410882651805878 Conv_1_aggregated_features[4,13]
   - 1.855417490005493 Conv_1_aggregated_features[4,14]
   + 0.247613787651062 Conv_1_aggregated_features[4,15] + Conv_1_t[4,14]
   = -0.2306040376424789
 Conv_1_output_constraint[4,15]: 0.043729841709137 Conv_0_Relu_ts[4,0]
   + 0.045797374099493 Conv_0_Relu_ts[4,1]
   - 0.1762047111988068 Conv_0_Relu_ts[4,2]
   - 0.2225021123886108 Conv_0_Relu_ts[4,3]
   + 0.1344186365604401 Conv_0_Relu_ts[4,4]
   + 0.2975046932697296 Conv_0_Relu_ts[4,5]
   + 0.1135610193014145 Conv_0_Relu_ts[4,6]
   + 0.165085032582283 Conv_0_Relu_ts[4,7]
   + 0.0632660016417503 Conv_0_Relu_ts[4,8]
   + 0.2534148395061493 Conv_0_Relu_ts[4,9]
   + 0.1100017726421356 Conv_0_Relu_ts[4,10]
   + 0.045901358127594 Conv_0_Relu_ts[4,11]
   + 0.2102706432342529 Conv_0_Relu_ts[4,12]
   + 0.1767551600933075 Conv_0_Relu_ts[4,13]
   - 0.0864512398838997 Conv_0_Relu_ts[4,14]
   + 0.1616824418306351 Conv_0_Relu_ts[4,15]
   + 0.0113421082496643 Conv_1_aggregated_features[4,0]
   + 0.2387615889310837 Conv_1_aggregated_features[4,1]
   + 0.1763307154178619 Conv_1_aggregated_features[4,2]
   + 0.1442219316959381 Conv_1_aggregated_features[4,3]
   + 0.3573774099349976 Conv_1_aggregated_features[4,4]
   - 0.0406427308917046 Conv_1_aggregated_features[4,5]
   + 0.1155626475811005 Conv_1_aggregated_features[4,6]
   + 0.0186243671923876 Conv_1_aggregated_features[4,7]
   - 0.1389756500720978 Conv_1_aggregated_features[4,8]
   + 0.274962306022644 Conv_1_aggregated_features[4,9]
   + 0.0309249386191368 Conv_1_aggregated_features[4,10]
   - 0.1606657803058624 Conv_1_aggregated_features[4,11]
   + 0.2456246316432953 Conv_1_aggregated_features[4,12]
   + 0.4163841009140015 Conv_1_aggregated_features[4,13]
   + 0.29454505443573 Conv_1_aggregated_features[4,14]
   + 0.1688967496156693 Conv_1_aggregated_features[4,15] + Conv_1_t[4,15]
   = -0.112743653357029
 Conv_1_output_constraint[5,0]: - 0.3061666786670685 Conv_0_Relu_ts[5,0]
   + 0.8854666948318481 Conv_0_Relu_ts[5,1]
   - 1.108363628387451 Conv_0_Relu_ts[5,2]
   - 0.0485447347164154 Conv_0_Relu_ts[5,3]
   - 0.3054571747779846 Conv_0_Relu_ts[5,4]
   - 1.666001558303833 Conv_0_Relu_ts[5,5]
   + 1.575538277626038 Conv_0_Relu_ts[5,6]
   - 0.8786032199859619 Conv_0_Relu_ts[5,7]
   + 0.1696695387363434 Conv_0_Relu_ts[5,8]
   - 0.2058112174272537 Conv_0_Relu_ts[5,9]
   - 0.8619717359542847 Conv_0_Relu_ts[5,10]
   + 1.268710613250732 Conv_0_Relu_ts[5,11]
   + 0.1005195453763008 Conv_0_Relu_ts[5,12]
   - 0.00627355137839913 Conv_0_Relu_ts[5,13]
   - 0.6727985143661499 Conv_0_Relu_ts[5,14]
   - 0.6082888841629028 Conv_0_Relu_ts[5,15]
   + 0.0746944323182106 Conv_1_aggregated_features[5,0]
   - 0.5317568182945251 Conv_1_aggregated_features[5,1]
   + 0.4405037760734558 Conv_1_aggregated_features[5,2]
   - 0.1488203704357147 Conv_1_aggregated_features[5,3]
   + 2.01956033706665 Conv_1_aggregated_features[5,4]
   + 0.8369019031524658 Conv_1_aggregated_features[5,5]
   + 0.1561367213726044 Conv_1_aggregated_features[5,6]
   + 0.1698544919490814 Conv_1_aggregated_features[5,7]
   - 0.0660476386547089 Conv_1_aggregated_features[5,8]
   + 1.570046067237854 Conv_1_aggregated_features[5,9]
   + 0.5540977120399475 Conv_1_aggregated_features[5,10]
   - 1.275410532951355 Conv_1_aggregated_features[5,11]
   - 0.092900924384594 Conv_1_aggregated_features[5,12]
   + 3.287018537521362 Conv_1_aggregated_features[5,13]
   + 2.392175912857056 Conv_1_aggregated_features[5,14]
   + 0.3933185040950775 Conv_1_aggregated_features[5,15] + Conv_1_t[5,0]
   = -2.192610263824463
 Conv_1_output_constraint[5,1]: 0.5202509164810181 Conv_0_Relu_ts[5,0]
   + 0.6686695218086243 Conv_0_Relu_ts[5,1]
   - 0.4720649719238281 Conv_0_Relu_ts[5,2]
   + 0.1026173532009125 Conv_0_Relu_ts[5,3]
   + 0.2593795359134674 Conv_0_Relu_ts[5,4]
   - 2.458898544311523 Conv_0_Relu_ts[5,5]
   - 0.5632196664810181 Conv_0_Relu_ts[5,6]
   - 1.791601657867432 Conv_0_Relu_ts[5,7]
   - 0.1268838942050934 Conv_0_Relu_ts[5,8]
   + 0.7120329737663269 Conv_0_Relu_ts[5,9]
   - 0.5212016701698303 Conv_0_Relu_ts[5,10]
   + 0.6136131286621094 Conv_0_Relu_ts[5,11]
   + 1.563590168952942 Conv_0_Relu_ts[5,12]
   - 0.3512922525405884 Conv_0_Relu_ts[5,13]
   + 0.5421575903892517 Conv_0_Relu_ts[5,14]
   + 0.7331249117851257 Conv_0_Relu_ts[5,15]
   + 0.0113700022920966 Conv_1_aggregated_features[5,0]
   + 2.169116258621216 Conv_1_aggregated_features[5,1]
   + 0.00623909616842866 Conv_1_aggregated_features[5,2]
   + 0.0937856733798981 Conv_1_aggregated_features[5,3]
   + 0.8328568339347839 Conv_1_aggregated_features[5,4]
   + 0.383074939250946 Conv_1_aggregated_features[5,5]
   - 0.1267539262771606 Conv_1_aggregated_features[5,6]
   + 0.6484018564224243 Conv_1_aggregated_features[5,7]
   + 0.1870926916599274 Conv_1_aggregated_features[5,8]
   - 0.4116691052913666 Conv_1_aggregated_features[5,9]
   + 0.1439376026391983 Conv_1_aggregated_features[5,10]
   + 1.05384635925293 Conv_1_aggregated_features[5,11]
   + 1.155439853668213 Conv_1_aggregated_features[5,12]
   - 0.2060132175683975 Conv_1_aggregated_features[5,13]
   + 0.6003169417381287 Conv_1_aggregated_features[5,14]
   + 0.1458597332239151 Conv_1_aggregated_features[5,15] + Conv_1_t[5,1]
   = -0.1077260375022888
 Conv_1_output_constraint[5,2]: 0.4269110858440399 Conv_0_Relu_ts[5,0]
   - 1.097789287567139 Conv_0_Relu_ts[5,1]
   - 0.3380239307880402 Conv_0_Relu_ts[5,2]
   + 0.23756343126297 Conv_0_Relu_ts[5,3]
   - 0.337738424539566 Conv_0_Relu_ts[5,4]
   - 1.407688736915588 Conv_0_Relu_ts[5,5]
   - 0.7854894995689392 Conv_0_Relu_ts[5,6]
   - 1.080899715423584 Conv_0_Relu_ts[5,7]
   + 0.0233876463025808 Conv_0_Relu_ts[5,8]
   + 1.849825739860535 Conv_0_Relu_ts[5,9]
   + 0.4580322206020355 Conv_0_Relu_ts[5,10]
   - 0.819361686706543 Conv_0_Relu_ts[5,11]
   - 0.0822571739554405 Conv_0_Relu_ts[5,12]
   - 0.1501271426677704 Conv_0_Relu_ts[5,13]
   + 0.1831801384687424 Conv_0_Relu_ts[5,14]
   + 0.00308510870672762 Conv_0_Relu_ts[5,15]
   - 0.2153502702713013 Conv_1_aggregated_features[5,0]
   + 3.817557334899902 Conv_1_aggregated_features[5,1]
   - 0.3402683734893799 Conv_1_aggregated_features[5,2]
   + 0.0650523602962494 Conv_1_aggregated_features[5,3]
   - 0.4435569643974304 Conv_1_aggregated_features[5,4]
   + 0.045248981565237 Conv_1_aggregated_features[5,5]
   + 0.844337522983551 Conv_1_aggregated_features[5,6]
   - 0.0691490024328232 Conv_1_aggregated_features[5,7]
   + 0.0830505788326263 Conv_1_aggregated_features[5,8]
   - 0.3022302687168121 Conv_1_aggregated_features[5,9]
   - 0.611566960811615 Conv_1_aggregated_features[5,10]
   + 3.582128524780273 Conv_1_aggregated_features[5,11]
   + 0.1314867585897446 Conv_1_aggregated_features[5,12]
   + 0.5333523154258728 Conv_1_aggregated_features[5,13]
   - 0.1983102858066559 Conv_1_aggregated_features[5,14]
   + 0.1628565639257431 Conv_1_aggregated_features[5,15] + Conv_1_t[5,2]
   = -0.4443503022193909
 Conv_1_output_constraint[5,3]: 0.066770151257515 Conv_0_Relu_ts[5,0]
   + 0.3136637210845947 Conv_0_Relu_ts[5,1]
   + 0.2078424692153931 Conv_0_Relu_ts[5,2]
   - 0.1673517823219299 Conv_0_Relu_ts[5,3]
   + 1.022224068641663 Conv_0_Relu_ts[5,4]
   - 0.1723377108573914 Conv_0_Relu_ts[5,5]
   + 0.3032898306846619 Conv_0_Relu_ts[5,6]
   + 0.0841449126601219 Conv_0_Relu_ts[5,7]
   + 0.0921086072921753 Conv_0_Relu_ts[5,8]
   - 0.022469462826848 Conv_0_Relu_ts[5,9]
   - 0.1657154262065887 Conv_0_Relu_ts[5,10]
   + 1.007031679153442 Conv_0_Relu_ts[5,11]
   + 0.0208180602639914 Conv_0_Relu_ts[5,12]
   + 0.3933880031108856 Conv_0_Relu_ts[5,13]
   + 0.3077191412448883 Conv_0_Relu_ts[5,14]
   + 0.0400979965925217 Conv_0_Relu_ts[5,15]
   + 0.0230858232825994 Conv_1_aggregated_features[5,0]
   + 0.3792195022106171 Conv_1_aggregated_features[5,1]
   + 0.1572305560112 Conv_1_aggregated_features[5,2]
   - 0.0165807008743286 Conv_1_aggregated_features[5,3]
   + 0.5649795532226562 Conv_1_aggregated_features[5,4]
   + 0.6253551840782166 Conv_1_aggregated_features[5,5]
   + 0.3968201577663422 Conv_1_aggregated_features[5,6]
   + 0.2728661298751831 Conv_1_aggregated_features[5,7]
   - 0.210608571767807 Conv_1_aggregated_features[5,8]
   + 0.094737209379673 Conv_1_aggregated_features[5,9]
   - 0.1354089975357056 Conv_1_aggregated_features[5,10]
   + 0.9330839514732361 Conv_1_aggregated_features[5,11]
   + 0.1677136421203613 Conv_1_aggregated_features[5,12]
   + 0.6573846936225891 Conv_1_aggregated_features[5,13]
   + 0.044663343578577 Conv_1_aggregated_features[5,14]
   - 0.0132347755134106 Conv_1_aggregated_features[5,15] + Conv_1_t[5,3]
   = -0.327608048915863
 Conv_1_output_constraint[5,4]: - 0.0720538049936295 Conv_0_Relu_ts[5,0]
   + 3.730369091033936 Conv_0_Relu_ts[5,1]
   - 0.8118388652801514 Conv_0_Relu_ts[5,2]
   - 0.174347460269928 Conv_0_Relu_ts[5,3]
   - 3.084089279174805 Conv_0_Relu_ts[5,4]
   - 5.146248340606689 Conv_0_Relu_ts[5,5]
   - 2.440467357635498 Conv_0_Relu_ts[5,6]
   - 3.765322923660278 Conv_0_Relu_ts[5,7]
   + 0.153628796339035 Conv_0_Relu_ts[5,8]
   + 0.1013353765010834 Conv_0_Relu_ts[5,9]
   - 0.5803871154785156 Conv_0_Relu_ts[5,10]
   - 2.657181739807129 Conv_0_Relu_ts[5,11]
   + 0.6840037703514099 Conv_0_Relu_ts[5,12]
   - 1.184461355209351 Conv_0_Relu_ts[5,13]
   - 0.3772841989994049 Conv_0_Relu_ts[5,14]
   + 0.5856359601020813 Conv_0_Relu_ts[5,15]
   + 0.2062403708696365 Conv_1_aggregated_features[5,0]
   + 1.470675468444824 Conv_1_aggregated_features[5,1]
   + 0.3523006737232208 Conv_1_aggregated_features[5,2]
   - 0.1351108849048615 Conv_1_aggregated_features[5,3]
   + 0.8932182788848877 Conv_1_aggregated_features[5,4]
   + 1.194340229034424 Conv_1_aggregated_features[5,5]
   + 0.0163087919354439 Conv_1_aggregated_features[5,6]
   - 0.5082771182060242 Conv_1_aggregated_features[5,7]
   - 0.0201514065265656 Conv_1_aggregated_features[5,8]
   + 0.9045064449310303 Conv_1_aggregated_features[5,9]
   + 0.7243757843971252 Conv_1_aggregated_features[5,10]
   + 0.5823414921760559 Conv_1_aggregated_features[5,11]
   + 0.0738087296485901 Conv_1_aggregated_features[5,12]
   - 0.3080458641052246 Conv_1_aggregated_features[5,13]
   + 0.9457569122314453 Conv_1_aggregated_features[5,14]
   + 0.4423830211162567 Conv_1_aggregated_features[5,15] + Conv_1_t[5,4]
   = -0.2138544023036957
 Conv_1_output_constraint[5,5]: 0.2116706073284149 Conv_0_Relu_ts[5,0]
   + 0.2319677472114563 Conv_0_Relu_ts[5,1]
   + 0.1967353820800781 Conv_0_Relu_ts[5,2]
   - 0.1974936425685883 Conv_0_Relu_ts[5,3]
   + 0.087493434548378 Conv_0_Relu_ts[5,4]
   + 0.0226023308932781 Conv_0_Relu_ts[5,5]
   - 0.1455883532762527 Conv_0_Relu_ts[5,6]
   - 0.1098284423351288 Conv_0_Relu_ts[5,7]
   + 0.0375414304435253 Conv_0_Relu_ts[5,8]
   + 0.2209681868553162 Conv_0_Relu_ts[5,9]
   + 0.0180207490921021 Conv_0_Relu_ts[5,10]
   - 0.0944192707538605 Conv_0_Relu_ts[5,11]
   + 0.1318628787994385 Conv_0_Relu_ts[5,12]
   + 0.1718421876430511 Conv_0_Relu_ts[5,13]
   - 0.0992768108844757 Conv_0_Relu_ts[5,14]
   + 0.2220577001571655 Conv_0_Relu_ts[5,15]
   + 0.0500434041023254 Conv_1_aggregated_features[5,0]
   + 0.2417693138122559 Conv_1_aggregated_features[5,1]
   + 0.1884889900684357 Conv_1_aggregated_features[5,2]
   - 0.0459566116333008 Conv_1_aggregated_features[5,3]
   - 0.1095028519630432 Conv_1_aggregated_features[5,4]
   - 0.1494371891021729 Conv_1_aggregated_features[5,5]
   - 0.0601528584957123 Conv_1_aggregated_features[5,6]
   - 0.2431226968765259 Conv_1_aggregated_features[5,7]
   + 0.0709281861782074 Conv_1_aggregated_features[5,8]
   + 0.1859873235225677 Conv_1_aggregated_features[5,9]
   - 0.0025022029876709 Conv_1_aggregated_features[5,10]
   + 0.1165908575057983 Conv_1_aggregated_features[5,11]
   - 0.0946725904941559 Conv_1_aggregated_features[5,12]
   + 0.1679102778434753 Conv_1_aggregated_features[5,13]
   + 0.0888539850711823 Conv_1_aggregated_features[5,14]
   + 0.0481749773025513 Conv_1_aggregated_features[5,15] + Conv_1_t[5,5]
   = -0.2017861604690552
 Conv_1_output_constraint[5,6]: - 0.5161762237548828 Conv_0_Relu_ts[5,0]
   - 0.9039249420166016 Conv_0_Relu_ts[5,1]
   - 0.5091935992240906 Conv_0_Relu_ts[5,2]
   + 0.177293062210083 Conv_0_Relu_ts[5,3]
   + 1.534894466400146 Conv_0_Relu_ts[5,4]
   + 0.3099820613861084 Conv_0_Relu_ts[5,5]
   + 1.335353136062622 Conv_0_Relu_ts[5,6]
   - 0.3965733051300049 Conv_0_Relu_ts[5,7]
   + 0.0226722583174706 Conv_0_Relu_ts[5,8]
   - 0.0153551828116179 Conv_0_Relu_ts[5,9]
   - 0.188335582613945 Conv_0_Relu_ts[5,10]
   + 0.1465377509593964 Conv_0_Relu_ts[5,11]
   - 0.4378408491611481 Conv_0_Relu_ts[5,12]
   + 0.5950998067855835 Conv_0_Relu_ts[5,13]
   - 0.2773410975933075 Conv_0_Relu_ts[5,14]
   - 0.3855878710746765 Conv_0_Relu_ts[5,15]
   + 0.0145458607003093 Conv_1_aggregated_features[5,0]
   - 1.349490284919739 Conv_1_aggregated_features[5,1]
   - 0.6738612055778503 Conv_1_aggregated_features[5,2]
   - 0.1977581083774567 Conv_1_aggregated_features[5,3]
   + 3.616869688034058 Conv_1_aggregated_features[5,4]
   - 0.5846084952354431 Conv_1_aggregated_features[5,5]
   + 0.3911246955394745 Conv_1_aggregated_features[5,6]
   - 0.2603789269924164 Conv_1_aggregated_features[5,7]
   + 0.0530429184436798 Conv_1_aggregated_features[5,8]
   + 0.7039647698402405 Conv_1_aggregated_features[5,9]
   - 0.4948081970214844 Conv_1_aggregated_features[5,10]
   - 1.33747124671936 Conv_1_aggregated_features[5,11]
   + 0.4521820247173309 Conv_1_aggregated_features[5,12]
   + 2.879487276077271 Conv_1_aggregated_features[5,13]
   + 0.6494181156158447 Conv_1_aggregated_features[5,14]
   - 0.0441654324531555 Conv_1_aggregated_features[5,15] + Conv_1_t[5,6]
   = -1.141431212425232
 Conv_1_output_constraint[5,7]: 0.1232976838946342 Conv_0_Relu_ts[5,0]
   + 0.7059634327888489 Conv_0_Relu_ts[5,1]
   + 0.0986589044332504 Conv_0_Relu_ts[5,2]
   - 0.1159188747406006 Conv_0_Relu_ts[5,3]
   + 0.1207201555371284 Conv_0_Relu_ts[5,4]
   + 0.4950260519981384 Conv_0_Relu_ts[5,5]
   - 0.1985135078430176 Conv_0_Relu_ts[5,6]
   + 0.3846564292907715 Conv_0_Relu_ts[5,7]
   - 0.2091868817806244 Conv_0_Relu_ts[5,8]
   + 0.0282646510750055 Conv_0_Relu_ts[5,9]
   + 0.00616153934970498 Conv_0_Relu_ts[5,10]
   - 0.2992357313632965 Conv_0_Relu_ts[5,11]
   + 0.6717308759689331 Conv_0_Relu_ts[5,12]
   + 0.3884595632553101 Conv_0_Relu_ts[5,13]
   + 0.1088093742728233 Conv_0_Relu_ts[5,14]
   + 0.1363743990659714 Conv_0_Relu_ts[5,15]
   + 0.2181874513626099 Conv_1_aggregated_features[5,0]
   + 0.1884859055280685 Conv_1_aggregated_features[5,1]
   + 0.1986039727926254 Conv_1_aggregated_features[5,2]
   - 0.0569350719451904 Conv_1_aggregated_features[5,3]
   + 0.4957895576953888 Conv_1_aggregated_features[5,4]
   - 0.2215360552072525 Conv_1_aggregated_features[5,5]
   - 0.3126825392246246 Conv_1_aggregated_features[5,6]
   + 0.2129529714584351 Conv_1_aggregated_features[5,7]
   + 0.1905750632286072 Conv_1_aggregated_features[5,8]
   + 0.3835417926311493 Conv_1_aggregated_features[5,9]
   + 0.0515264272689819 Conv_1_aggregated_features[5,10]
   + 0.2774861454963684 Conv_1_aggregated_features[5,11]
   + 0.1720549166202545 Conv_1_aggregated_features[5,12]
   + 0.2042496353387833 Conv_1_aggregated_features[5,13]
   + 0.1717791706323624 Conv_1_aggregated_features[5,14]
   - 0.0635386854410172 Conv_1_aggregated_features[5,15] + Conv_1_t[5,7]
   = -0.1961010098457336
 Conv_1_output_constraint[5,8]: - 0.0984712317585945 Conv_0_Relu_ts[5,0]
   - 0.8972246050834656 Conv_0_Relu_ts[5,1]
   - 0.1342054009437561 Conv_0_Relu_ts[5,2]
   + 0.0934775173664093 Conv_0_Relu_ts[5,3]
   + 3.929172039031982 Conv_0_Relu_ts[5,4]
   - 1.699812531471252 Conv_0_Relu_ts[5,5]
   + 0.4780299663543701 Conv_0_Relu_ts[5,6]
   + 0.1916014850139618 Conv_0_Relu_ts[5,7]
   - 0.0282735247164965 Conv_0_Relu_ts[5,8]
   - 0.10334312915802 Conv_0_Relu_ts[5,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[5,10]
   - 0.9000989198684692 Conv_0_Relu_ts[5,11]
   - 0.3381493389606476 Conv_0_Relu_ts[5,12]
   + 0.2327869236469269 Conv_0_Relu_ts[5,13]
   + 0.6162272691726685 Conv_0_Relu_ts[5,14]
   - 0.1045716181397438 Conv_0_Relu_ts[5,15]
   - 0.0895406752824783 Conv_1_aggregated_features[5,0]
   - 0.0733596011996269 Conv_1_aggregated_features[5,1]
   - 0.6064443588256836 Conv_1_aggregated_features[5,2]
   + 0.1685465276241302 Conv_1_aggregated_features[5,3]
   + 2.521227836608887 Conv_1_aggregated_features[5,4]
   + 0.8868938088417053 Conv_1_aggregated_features[5,5]
   - 0.3730944097042084 Conv_1_aggregated_features[5,6]
   + 0.4433300495147705 Conv_1_aggregated_features[5,7]
   + 0.2465403079986572 Conv_1_aggregated_features[5,8]
   + 0.1199371814727783 Conv_1_aggregated_features[5,9]
   - 0.427333652973175 Conv_1_aggregated_features[5,10]
   - 1.286244511604309 Conv_1_aggregated_features[5,11]
   + 0.4271916449069977 Conv_1_aggregated_features[5,12]
   + 1.475327014923096 Conv_1_aggregated_features[5,13]
   - 0.2351342439651489 Conv_1_aggregated_features[5,14]
   - 0.1273849010467529 Conv_1_aggregated_features[5,15] + Conv_1_t[5,8]
   = -1.303357601165771
 Conv_1_output_constraint[5,9]: 0.1534582376480103 Conv_0_Relu_ts[5,0]
   + 1.456407189369202 Conv_0_Relu_ts[5,1]
   - 0.0228729248046875 Conv_0_Relu_ts[5,2]
   + 0.1227743625640869 Conv_0_Relu_ts[5,3]
   + 0.7793847918510437 Conv_0_Relu_ts[5,4]
   - 0.2757052779197693 Conv_0_Relu_ts[5,5]
   + 0.6606881022453308 Conv_0_Relu_ts[5,6]
   - 1.107266664505005 Conv_0_Relu_ts[5,7]
   + 0.1723826229572296 Conv_0_Relu_ts[5,8]
   - 0.5540673732757568 Conv_0_Relu_ts[5,9]
   + 0.3133971095085144 Conv_0_Relu_ts[5,10]
   + 0.9318023324012756 Conv_0_Relu_ts[5,11]
   + 0.5855908989906311 Conv_0_Relu_ts[5,12]
   - 0.2116718590259552 Conv_0_Relu_ts[5,13]
   - 0.3693177700042725 Conv_0_Relu_ts[5,14]
   - 0.0642165616154671 Conv_0_Relu_ts[5,15]
   + 0.1819062232971191 Conv_1_aggregated_features[5,0]
   + 1.311912417411804 Conv_1_aggregated_features[5,1]
   + 0.3460695445537567 Conv_1_aggregated_features[5,2]
   + 0.1086616516113281 Conv_1_aggregated_features[5,3]
   - 0.6099227070808411 Conv_1_aggregated_features[5,4]
   + 0.3786990940570831 Conv_1_aggregated_features[5,5]
   + 0.9473716020584106 Conv_1_aggregated_features[5,6]
   + 0.2196423262357712 Conv_1_aggregated_features[5,7]
   + 0.0830722749233246 Conv_1_aggregated_features[5,8]
   - 0.4822347462177277 Conv_1_aggregated_features[5,9]
   + 0.5108910202980042 Conv_1_aggregated_features[5,10]
   + 3.214276790618896 Conv_1_aggregated_features[5,11]
   - 0.7224436402320862 Conv_1_aggregated_features[5,12]
   + 0.2678262591362 Conv_1_aggregated_features[5,13]
   - 0.8807565569877625 Conv_1_aggregated_features[5,14]
   - 0.3268090188503265 Conv_1_aggregated_features[5,15] + Conv_1_t[5,9]
   = -0.8283404111862183
 Conv_1_output_constraint[5,10]: 0.3653882741928101 Conv_0_Relu_ts[5,0]
   - 0.8208209872245789 Conv_0_Relu_ts[5,1]
   + 0.3088222146034241 Conv_0_Relu_ts[5,2]
   + 0.1173163652420044 Conv_0_Relu_ts[5,3]
   + 0.658578634262085 Conv_0_Relu_ts[5,4]
   + 0.584019660949707 Conv_0_Relu_ts[5,5]
   - 0.8654046058654785 Conv_0_Relu_ts[5,6]
   + 0.3697350919246674 Conv_0_Relu_ts[5,7]
   - 0.2274059355258942 Conv_0_Relu_ts[5,8]
   + 1.197938442230225 Conv_0_Relu_ts[5,9]
   + 0.2446341812610626 Conv_0_Relu_ts[5,10]
   - 0.5676743984222412 Conv_0_Relu_ts[5,11]
   - 0.3123646676540375 Conv_0_Relu_ts[5,12]
   + 3.181503534317017 Conv_0_Relu_ts[5,13]
   + 0.6926240921020508 Conv_0_Relu_ts[5,14]
   + 0.0886927545070648 Conv_0_Relu_ts[5,15]
   - 0.2005636394023895 Conv_1_aggregated_features[5,0]
   + 1.403087973594666 Conv_1_aggregated_features[5,1]
   + 0.0523448958992958 Conv_1_aggregated_features[5,2]
   + 0.2411543428897858 Conv_1_aggregated_features[5,3]
   + 0.5953171253204346 Conv_1_aggregated_features[5,4]
   - 0.8278964757919312 Conv_1_aggregated_features[5,5]
   + 0.2499217540025711 Conv_1_aggregated_features[5,6]
   - 0.7393644452095032 Conv_1_aggregated_features[5,7]
   + 0.0902875959873199 Conv_1_aggregated_features[5,8]
   + 0.1488711386919022 Conv_1_aggregated_features[5,9]
   - 0.0303693022578955 Conv_1_aggregated_features[5,10]
   + 2.024244070053101 Conv_1_aggregated_features[5,11]
   + 0.495431512594223 Conv_1_aggregated_features[5,12]
   + 0.3463623225688934 Conv_1_aggregated_features[5,13]
   - 0.0763808861374855 Conv_1_aggregated_features[5,14]
   + 0.1597562879323959 Conv_1_aggregated_features[5,15] + Conv_1_t[5,10]
   = -0.5063248872756958
 Conv_1_output_constraint[5,11]: - 0.5185666680335999 Conv_0_Relu_ts[5,0]
   + 3.656918525695801 Conv_0_Relu_ts[5,1]
   - 0.7082476615905762 Conv_0_Relu_ts[5,2]
   + 0.1922211349010468 Conv_0_Relu_ts[5,3]
   - 0.7149283289909363 Conv_0_Relu_ts[5,4]
   - 3.255658626556396 Conv_0_Relu_ts[5,5]
   + 2.464392185211182 Conv_0_Relu_ts[5,6]
   - 0.9477378129959106 Conv_0_Relu_ts[5,7]
   + 0.168237030506134 Conv_0_Relu_ts[5,8]
   - 1.177640557289124 Conv_0_Relu_ts[5,9]
   - 0.7062549591064453 Conv_0_Relu_ts[5,10]
   + 1.453806400299072 Conv_0_Relu_ts[5,11]
   + 0.2612711191177368 Conv_0_Relu_ts[5,12]
   + 0.2999382615089417 Conv_0_Relu_ts[5,13]
   - 1.255579590797424 Conv_0_Relu_ts[5,14]
   - 0.4265820384025574 Conv_0_Relu_ts[5,15]
   + 0.3379200994968414 Conv_1_aggregated_features[5,0]
   - 1.33229672908783 Conv_1_aggregated_features[5,1]
   - 0.1594242304563522 Conv_1_aggregated_features[5,2]
   + 0.2298935651779175 Conv_1_aggregated_features[5,3]
   - 1.311494588851929 Conv_1_aggregated_features[5,4]
   + 3.605161666870117 Conv_1_aggregated_features[5,5]
   - 0.3176752626895905 Conv_1_aggregated_features[5,6]
   + 0.3926282525062561 Conv_1_aggregated_features[5,7]
   + 0.1479305028915405 Conv_1_aggregated_features[5,8]
   - 0.0372937023639679 Conv_1_aggregated_features[5,9]
   + 0.3455427587032318 Conv_1_aggregated_features[5,10]
   - 2.002299070358276 Conv_1_aggregated_features[5,11]
   - 0.9904713034629822 Conv_1_aggregated_features[5,12]
   + 0.0751789212226868 Conv_1_aggregated_features[5,13]
   - 0.5166233777999878 Conv_1_aggregated_features[5,14]
   + 0.0510688796639442 Conv_1_aggregated_features[5,15] + Conv_1_t[5,11]
   = -1.480785727500916
 Conv_1_output_constraint[5,12]: 0.3080185949802399 Conv_0_Relu_ts[5,0]
   + 0.0994751453399658 Conv_0_Relu_ts[5,1]
   + 0.4488444924354553 Conv_0_Relu_ts[5,2]
   + 0.0305773317813873 Conv_0_Relu_ts[5,3]
   - 1.673393130302429 Conv_0_Relu_ts[5,4]
   + 0.2824500501155853 Conv_0_Relu_ts[5,5]
   - 1.192709684371948 Conv_0_Relu_ts[5,6]
   + 1.454243063926697 Conv_0_Relu_ts[5,7]
   + 0.1514801830053329 Conv_0_Relu_ts[5,8]
   + 1.026351928710938 Conv_0_Relu_ts[5,9]
   + 0.1183762699365616 Conv_0_Relu_ts[5,10]
   + 0.352224588394165 Conv_0_Relu_ts[5,11]
   + 1.090276837348938 Conv_0_Relu_ts[5,12]
   - 2.053535461425781 Conv_0_Relu_ts[5,13]
   + 2.538641452789307 Conv_0_Relu_ts[5,14]
   + 0.6387560963630676 Conv_0_Relu_ts[5,15]
   - 0.1990496516227722 Conv_1_aggregated_features[5,0]
   + 1.247365713119507 Conv_1_aggregated_features[5,1]
   + 0.2213435918092728 Conv_1_aggregated_features[5,2]
   + 0.2398849725723267 Conv_1_aggregated_features[5,3]
   - 2.001479864120483 Conv_1_aggregated_features[5,4]
   - 1.084800243377686 Conv_1_aggregated_features[5,5]
   + 0.4287458956241608 Conv_1_aggregated_features[5,6]
   - 0.8179780840873718 Conv_1_aggregated_features[5,7]
   - 0.0883614718914032 Conv_1_aggregated_features[5,8]
   - 0.4398066997528076 Conv_1_aggregated_features[5,9]
   + 0.2760448455810547 Conv_1_aggregated_features[5,10]
   + 1.944519639015198 Conv_1_aggregated_features[5,11]
   + 1.041437149047852 Conv_1_aggregated_features[5,12]
   - 1.682050704956055 Conv_1_aggregated_features[5,13]
   - 0.1825132220983505 Conv_1_aggregated_features[5,14]
   + 0.2881160080432892 Conv_1_aggregated_features[5,15] + Conv_1_t[5,12]
   = 2.457676649093628
 Conv_1_output_constraint[5,13]: 0.2853958606719971 Conv_0_Relu_ts[5,0]
   + 0.0445350483059883 Conv_0_Relu_ts[5,1]
   - 0.0597315393388271 Conv_0_Relu_ts[5,2]
   + 0.2403226792812347 Conv_0_Relu_ts[5,3]
   + 0.5103238224983215 Conv_0_Relu_ts[5,4]
   - 0.1369927227497101 Conv_0_Relu_ts[5,5]
   + 0.5091087818145752 Conv_0_Relu_ts[5,6]
   - 0.0576926171779633 Conv_0_Relu_ts[5,7]
   - 0.00701320171356201 Conv_0_Relu_ts[5,8]
   + 0.5782948136329651 Conv_0_Relu_ts[5,9]
   - 0.1188536584377289 Conv_0_Relu_ts[5,10]
   - 0.078026682138443 Conv_0_Relu_ts[5,11]
   + 0.1965750753879547 Conv_0_Relu_ts[5,12]
   + 0.4982857704162598 Conv_0_Relu_ts[5,13]
   - 0.0995257869362831 Conv_0_Relu_ts[5,14]
   + 0.0712818428874016 Conv_0_Relu_ts[5,15]
   + 0.2628078460693359 Conv_1_aggregated_features[5,0]
   + 0.6577805280685425 Conv_1_aggregated_features[5,1]
   - 0.1415485441684723 Conv_1_aggregated_features[5,2]
   + 0.1597791016101837 Conv_1_aggregated_features[5,3]
   + 0.3969745337963104 Conv_1_aggregated_features[5,4]
   - 0.3941676616668701 Conv_1_aggregated_features[5,5]
   + 0.3484204709529877 Conv_1_aggregated_features[5,6]
   + 0.2601790428161621 Conv_1_aggregated_features[5,7]
   - 0.0652478933334351 Conv_1_aggregated_features[5,8]
   + 0.3492280542850494 Conv_1_aggregated_features[5,9]
   + 0.1917867064476013 Conv_1_aggregated_features[5,10]
   + 0.2439181953668594 Conv_1_aggregated_features[5,11]
   + 0.1009163707494736 Conv_1_aggregated_features[5,12]
   + 0.2350947707891464 Conv_1_aggregated_features[5,13]
   + 0.2627962231636047 Conv_1_aggregated_features[5,14]
   - 0.0959920063614845 Conv_1_aggregated_features[5,15] + Conv_1_t[5,13]
   = -0.4268372058868408
 Conv_1_output_constraint[5,14]: - 0.2448655962944031 Conv_0_Relu_ts[5,0]
   + 0.2688764035701752 Conv_0_Relu_ts[5,1]
   + 0.0118266940116882 Conv_0_Relu_ts[5,2]
   - 0.0670525431632996 Conv_0_Relu_ts[5,3]
   + 0.1511579602956772 Conv_0_Relu_ts[5,4]
   - 0.4796198606491089 Conv_0_Relu_ts[5,5]
   - 0.1699391901493073 Conv_0_Relu_ts[5,6]
   + 0.2647148370742798 Conv_0_Relu_ts[5,7]
   + 0.137117475271225 Conv_0_Relu_ts[5,8]
   + 0.1737731248140335 Conv_0_Relu_ts[5,9]
   - 0.1610838174819946 Conv_0_Relu_ts[5,10]
   - 0.6311220526695251 Conv_0_Relu_ts[5,11]
   + 0.6263757348060608 Conv_0_Relu_ts[5,12]
   - 0.4823384284973145 Conv_0_Relu_ts[5,13]
   + 0.2192765176296234 Conv_0_Relu_ts[5,14]
   + 0.1739131808280945 Conv_0_Relu_ts[5,15]
   + 0.0443203188478947 Conv_1_aggregated_features[5,0]
   + 0.7668815851211548 Conv_1_aggregated_features[5,1]
   + 0.248365581035614 Conv_1_aggregated_features[5,2]
   - 0.2433404624462128 Conv_1_aggregated_features[5,3]
   + 0.2770798802375793 Conv_1_aggregated_features[5,4]
   - 1.224791765213013 Conv_1_aggregated_features[5,5]
   - 0.1892879456281662 Conv_1_aggregated_features[5,6]
   + 0.1141589805483818 Conv_1_aggregated_features[5,7]
   + 0.2022489905357361 Conv_1_aggregated_features[5,8]
   - 0.0391396470367908 Conv_1_aggregated_features[5,9]
   - 0.0935317650437355 Conv_1_aggregated_features[5,10]
   - 0.5407891273498535 Conv_1_aggregated_features[5,11]
   + 0.3130630552768707 Conv_1_aggregated_features[5,12]
   + 0.3410882651805878 Conv_1_aggregated_features[5,13]
   - 1.855417490005493 Conv_1_aggregated_features[5,14]
   + 0.247613787651062 Conv_1_aggregated_features[5,15] + Conv_1_t[5,14]
   = -0.2306040376424789
 Conv_1_output_constraint[5,15]: 0.043729841709137 Conv_0_Relu_ts[5,0]
   + 0.045797374099493 Conv_0_Relu_ts[5,1]
   - 0.1762047111988068 Conv_0_Relu_ts[5,2]
   - 0.2225021123886108 Conv_0_Relu_ts[5,3]
   + 0.1344186365604401 Conv_0_Relu_ts[5,4]
   + 0.2975046932697296 Conv_0_Relu_ts[5,5]
   + 0.1135610193014145 Conv_0_Relu_ts[5,6]
   + 0.165085032582283 Conv_0_Relu_ts[5,7]
   + 0.0632660016417503 Conv_0_Relu_ts[5,8]
   + 0.2534148395061493 Conv_0_Relu_ts[5,9]
   + 0.1100017726421356 Conv_0_Relu_ts[5,10]
   + 0.045901358127594 Conv_0_Relu_ts[5,11]
   + 0.2102706432342529 Conv_0_Relu_ts[5,12]
   + 0.1767551600933075 Conv_0_Relu_ts[5,13]
   - 0.0864512398838997 Conv_0_Relu_ts[5,14]
   + 0.1616824418306351 Conv_0_Relu_ts[5,15]
   + 0.0113421082496643 Conv_1_aggregated_features[5,0]
   + 0.2387615889310837 Conv_1_aggregated_features[5,1]
   + 0.1763307154178619 Conv_1_aggregated_features[5,2]
   + 0.1442219316959381 Conv_1_aggregated_features[5,3]
   + 0.3573774099349976 Conv_1_aggregated_features[5,4]
   - 0.0406427308917046 Conv_1_aggregated_features[5,5]
   + 0.1155626475811005 Conv_1_aggregated_features[5,6]
   + 0.0186243671923876 Conv_1_aggregated_features[5,7]
   - 0.1389756500720978 Conv_1_aggregated_features[5,8]
   + 0.274962306022644 Conv_1_aggregated_features[5,9]
   + 0.0309249386191368 Conv_1_aggregated_features[5,10]
   - 0.1606657803058624 Conv_1_aggregated_features[5,11]
   + 0.2456246316432953 Conv_1_aggregated_features[5,12]
   + 0.4163841009140015 Conv_1_aggregated_features[5,13]
   + 0.29454505443573 Conv_1_aggregated_features[5,14]
   + 0.1688967496156693 Conv_1_aggregated_features[5,15] + Conv_1_t[5,15]
   = -0.112743653357029
 Conv_1_output_constraint[6,0]: - 0.3061666786670685 Conv_0_Relu_ts[6,0]
   + 0.8854666948318481 Conv_0_Relu_ts[6,1]
   - 1.108363628387451 Conv_0_Relu_ts[6,2]
   - 0.0485447347164154 Conv_0_Relu_ts[6,3]
   - 0.3054571747779846 Conv_0_Relu_ts[6,4]
   - 1.666001558303833 Conv_0_Relu_ts[6,5]
   + 1.575538277626038 Conv_0_Relu_ts[6,6]
   - 0.8786032199859619 Conv_0_Relu_ts[6,7]
   + 0.1696695387363434 Conv_0_Relu_ts[6,8]
   - 0.2058112174272537 Conv_0_Relu_ts[6,9]
   - 0.8619717359542847 Conv_0_Relu_ts[6,10]
   + 1.268710613250732 Conv_0_Relu_ts[6,11]
   + 0.1005195453763008 Conv_0_Relu_ts[6,12]
   - 0.00627355137839913 Conv_0_Relu_ts[6,13]
   - 0.6727985143661499 Conv_0_Relu_ts[6,14]
   - 0.6082888841629028 Conv_0_Relu_ts[6,15]
   + 0.0746944323182106 Conv_1_aggregated_features[6,0]
   - 0.5317568182945251 Conv_1_aggregated_features[6,1]
   + 0.4405037760734558 Conv_1_aggregated_features[6,2]
   - 0.1488203704357147 Conv_1_aggregated_features[6,3]
   + 2.01956033706665 Conv_1_aggregated_features[6,4]
   + 0.8369019031524658 Conv_1_aggregated_features[6,5]
   + 0.1561367213726044 Conv_1_aggregated_features[6,6]
   + 0.1698544919490814 Conv_1_aggregated_features[6,7]
   - 0.0660476386547089 Conv_1_aggregated_features[6,8]
   + 1.570046067237854 Conv_1_aggregated_features[6,9]
   + 0.5540977120399475 Conv_1_aggregated_features[6,10]
   - 1.275410532951355 Conv_1_aggregated_features[6,11]
   - 0.092900924384594 Conv_1_aggregated_features[6,12]
   + 3.287018537521362 Conv_1_aggregated_features[6,13]
   + 2.392175912857056 Conv_1_aggregated_features[6,14]
   + 0.3933185040950775 Conv_1_aggregated_features[6,15] + Conv_1_t[6,0]
   = -2.192610263824463
 Conv_1_output_constraint[6,1]: 0.5202509164810181 Conv_0_Relu_ts[6,0]
   + 0.6686695218086243 Conv_0_Relu_ts[6,1]
   - 0.4720649719238281 Conv_0_Relu_ts[6,2]
   + 0.1026173532009125 Conv_0_Relu_ts[6,3]
   + 0.2593795359134674 Conv_0_Relu_ts[6,4]
   - 2.458898544311523 Conv_0_Relu_ts[6,5]
   - 0.5632196664810181 Conv_0_Relu_ts[6,6]
   - 1.791601657867432 Conv_0_Relu_ts[6,7]
   - 0.1268838942050934 Conv_0_Relu_ts[6,8]
   + 0.7120329737663269 Conv_0_Relu_ts[6,9]
   - 0.5212016701698303 Conv_0_Relu_ts[6,10]
   + 0.6136131286621094 Conv_0_Relu_ts[6,11]
   + 1.563590168952942 Conv_0_Relu_ts[6,12]
   - 0.3512922525405884 Conv_0_Relu_ts[6,13]
   + 0.5421575903892517 Conv_0_Relu_ts[6,14]
   + 0.7331249117851257 Conv_0_Relu_ts[6,15]
   + 0.0113700022920966 Conv_1_aggregated_features[6,0]
   + 2.169116258621216 Conv_1_aggregated_features[6,1]
   + 0.00623909616842866 Conv_1_aggregated_features[6,2]
   + 0.0937856733798981 Conv_1_aggregated_features[6,3]
   + 0.8328568339347839 Conv_1_aggregated_features[6,4]
   + 0.383074939250946 Conv_1_aggregated_features[6,5]
   - 0.1267539262771606 Conv_1_aggregated_features[6,6]
   + 0.6484018564224243 Conv_1_aggregated_features[6,7]
   + 0.1870926916599274 Conv_1_aggregated_features[6,8]
   - 0.4116691052913666 Conv_1_aggregated_features[6,9]
   + 0.1439376026391983 Conv_1_aggregated_features[6,10]
   + 1.05384635925293 Conv_1_aggregated_features[6,11]
   + 1.155439853668213 Conv_1_aggregated_features[6,12]
   - 0.2060132175683975 Conv_1_aggregated_features[6,13]
   + 0.6003169417381287 Conv_1_aggregated_features[6,14]
   + 0.1458597332239151 Conv_1_aggregated_features[6,15] + Conv_1_t[6,1]
   = -0.1077260375022888
 Conv_1_output_constraint[6,2]: 0.4269110858440399 Conv_0_Relu_ts[6,0]
   - 1.097789287567139 Conv_0_Relu_ts[6,1]
   - 0.3380239307880402 Conv_0_Relu_ts[6,2]
   + 0.23756343126297 Conv_0_Relu_ts[6,3]
   - 0.337738424539566 Conv_0_Relu_ts[6,4]
   - 1.407688736915588 Conv_0_Relu_ts[6,5]
   - 0.7854894995689392 Conv_0_Relu_ts[6,6]
   - 1.080899715423584 Conv_0_Relu_ts[6,7]
   + 0.0233876463025808 Conv_0_Relu_ts[6,8]
   + 1.849825739860535 Conv_0_Relu_ts[6,9]
   + 0.4580322206020355 Conv_0_Relu_ts[6,10]
   - 0.819361686706543 Conv_0_Relu_ts[6,11]
   - 0.0822571739554405 Conv_0_Relu_ts[6,12]
   - 0.1501271426677704 Conv_0_Relu_ts[6,13]
   + 0.1831801384687424 Conv_0_Relu_ts[6,14]
   + 0.00308510870672762 Conv_0_Relu_ts[6,15]
   - 0.2153502702713013 Conv_1_aggregated_features[6,0]
   + 3.817557334899902 Conv_1_aggregated_features[6,1]
   - 0.3402683734893799 Conv_1_aggregated_features[6,2]
   + 0.0650523602962494 Conv_1_aggregated_features[6,3]
   - 0.4435569643974304 Conv_1_aggregated_features[6,4]
   + 0.045248981565237 Conv_1_aggregated_features[6,5]
   + 0.844337522983551 Conv_1_aggregated_features[6,6]
   - 0.0691490024328232 Conv_1_aggregated_features[6,7]
   + 0.0830505788326263 Conv_1_aggregated_features[6,8]
   - 0.3022302687168121 Conv_1_aggregated_features[6,9]
   - 0.611566960811615 Conv_1_aggregated_features[6,10]
   + 3.582128524780273 Conv_1_aggregated_features[6,11]
   + 0.1314867585897446 Conv_1_aggregated_features[6,12]
   + 0.5333523154258728 Conv_1_aggregated_features[6,13]
   - 0.1983102858066559 Conv_1_aggregated_features[6,14]
   + 0.1628565639257431 Conv_1_aggregated_features[6,15] + Conv_1_t[6,2]
   = -0.4443503022193909
 Conv_1_output_constraint[6,3]: 0.066770151257515 Conv_0_Relu_ts[6,0]
   + 0.3136637210845947 Conv_0_Relu_ts[6,1]
   + 0.2078424692153931 Conv_0_Relu_ts[6,2]
   - 0.1673517823219299 Conv_0_Relu_ts[6,3]
   + 1.022224068641663 Conv_0_Relu_ts[6,4]
   - 0.1723377108573914 Conv_0_Relu_ts[6,5]
   + 0.3032898306846619 Conv_0_Relu_ts[6,6]
   + 0.0841449126601219 Conv_0_Relu_ts[6,7]
   + 0.0921086072921753 Conv_0_Relu_ts[6,8]
   - 0.022469462826848 Conv_0_Relu_ts[6,9]
   - 0.1657154262065887 Conv_0_Relu_ts[6,10]
   + 1.007031679153442 Conv_0_Relu_ts[6,11]
   + 0.0208180602639914 Conv_0_Relu_ts[6,12]
   + 0.3933880031108856 Conv_0_Relu_ts[6,13]
   + 0.3077191412448883 Conv_0_Relu_ts[6,14]
   + 0.0400979965925217 Conv_0_Relu_ts[6,15]
   + 0.0230858232825994 Conv_1_aggregated_features[6,0]
   + 0.3792195022106171 Conv_1_aggregated_features[6,1]
   + 0.1572305560112 Conv_1_aggregated_features[6,2]
   - 0.0165807008743286 Conv_1_aggregated_features[6,3]
   + 0.5649795532226562 Conv_1_aggregated_features[6,4]
   + 0.6253551840782166 Conv_1_aggregated_features[6,5]
   + 0.3968201577663422 Conv_1_aggregated_features[6,6]
   + 0.2728661298751831 Conv_1_aggregated_features[6,7]
   - 0.210608571767807 Conv_1_aggregated_features[6,8]
   + 0.094737209379673 Conv_1_aggregated_features[6,9]
   - 0.1354089975357056 Conv_1_aggregated_features[6,10]
   + 0.9330839514732361 Conv_1_aggregated_features[6,11]
   + 0.1677136421203613 Conv_1_aggregated_features[6,12]
   + 0.6573846936225891 Conv_1_aggregated_features[6,13]
   + 0.044663343578577 Conv_1_aggregated_features[6,14]
   - 0.0132347755134106 Conv_1_aggregated_features[6,15] + Conv_1_t[6,3]
   = -0.327608048915863
 Conv_1_output_constraint[6,4]: - 0.0720538049936295 Conv_0_Relu_ts[6,0]
   + 3.730369091033936 Conv_0_Relu_ts[6,1]
   - 0.8118388652801514 Conv_0_Relu_ts[6,2]
   - 0.174347460269928 Conv_0_Relu_ts[6,3]
   - 3.084089279174805 Conv_0_Relu_ts[6,4]
   - 5.146248340606689 Conv_0_Relu_ts[6,5]
   - 2.440467357635498 Conv_0_Relu_ts[6,6]
   - 3.765322923660278 Conv_0_Relu_ts[6,7]
   + 0.153628796339035 Conv_0_Relu_ts[6,8]
   + 0.1013353765010834 Conv_0_Relu_ts[6,9]
   - 0.5803871154785156 Conv_0_Relu_ts[6,10]
   - 2.657181739807129 Conv_0_Relu_ts[6,11]
   + 0.6840037703514099 Conv_0_Relu_ts[6,12]
   - 1.184461355209351 Conv_0_Relu_ts[6,13]
   - 0.3772841989994049 Conv_0_Relu_ts[6,14]
   + 0.5856359601020813 Conv_0_Relu_ts[6,15]
   + 0.2062403708696365 Conv_1_aggregated_features[6,0]
   + 1.470675468444824 Conv_1_aggregated_features[6,1]
   + 0.3523006737232208 Conv_1_aggregated_features[6,2]
   - 0.1351108849048615 Conv_1_aggregated_features[6,3]
   + 0.8932182788848877 Conv_1_aggregated_features[6,4]
   + 1.194340229034424 Conv_1_aggregated_features[6,5]
   + 0.0163087919354439 Conv_1_aggregated_features[6,6]
   - 0.5082771182060242 Conv_1_aggregated_features[6,7]
   - 0.0201514065265656 Conv_1_aggregated_features[6,8]
   + 0.9045064449310303 Conv_1_aggregated_features[6,9]
   + 0.7243757843971252 Conv_1_aggregated_features[6,10]
   + 0.5823414921760559 Conv_1_aggregated_features[6,11]
   + 0.0738087296485901 Conv_1_aggregated_features[6,12]
   - 0.3080458641052246 Conv_1_aggregated_features[6,13]
   + 0.9457569122314453 Conv_1_aggregated_features[6,14]
   + 0.4423830211162567 Conv_1_aggregated_features[6,15] + Conv_1_t[6,4]
   = -0.2138544023036957
 Conv_1_output_constraint[6,5]: 0.2116706073284149 Conv_0_Relu_ts[6,0]
   + 0.2319677472114563 Conv_0_Relu_ts[6,1]
   + 0.1967353820800781 Conv_0_Relu_ts[6,2]
   - 0.1974936425685883 Conv_0_Relu_ts[6,3]
   + 0.087493434548378 Conv_0_Relu_ts[6,4]
   + 0.0226023308932781 Conv_0_Relu_ts[6,5]
   - 0.1455883532762527 Conv_0_Relu_ts[6,6]
   - 0.1098284423351288 Conv_0_Relu_ts[6,7]
   + 0.0375414304435253 Conv_0_Relu_ts[6,8]
   + 0.2209681868553162 Conv_0_Relu_ts[6,9]
   + 0.0180207490921021 Conv_0_Relu_ts[6,10]
   - 0.0944192707538605 Conv_0_Relu_ts[6,11]
   + 0.1318628787994385 Conv_0_Relu_ts[6,12]
   + 0.1718421876430511 Conv_0_Relu_ts[6,13]
   - 0.0992768108844757 Conv_0_Relu_ts[6,14]
   + 0.2220577001571655 Conv_0_Relu_ts[6,15]
   + 0.0500434041023254 Conv_1_aggregated_features[6,0]
   + 0.2417693138122559 Conv_1_aggregated_features[6,1]
   + 0.1884889900684357 Conv_1_aggregated_features[6,2]
   - 0.0459566116333008 Conv_1_aggregated_features[6,3]
   - 0.1095028519630432 Conv_1_aggregated_features[6,4]
   - 0.1494371891021729 Conv_1_aggregated_features[6,5]
   - 0.0601528584957123 Conv_1_aggregated_features[6,6]
   - 0.2431226968765259 Conv_1_aggregated_features[6,7]
   + 0.0709281861782074 Conv_1_aggregated_features[6,8]
   + 0.1859873235225677 Conv_1_aggregated_features[6,9]
   - 0.0025022029876709 Conv_1_aggregated_features[6,10]
   + 0.1165908575057983 Conv_1_aggregated_features[6,11]
   - 0.0946725904941559 Conv_1_aggregated_features[6,12]
   + 0.1679102778434753 Conv_1_aggregated_features[6,13]
   + 0.0888539850711823 Conv_1_aggregated_features[6,14]
   + 0.0481749773025513 Conv_1_aggregated_features[6,15] + Conv_1_t[6,5]
   = -0.2017861604690552
 Conv_1_output_constraint[6,6]: - 0.5161762237548828 Conv_0_Relu_ts[6,0]
   - 0.9039249420166016 Conv_0_Relu_ts[6,1]
   - 0.5091935992240906 Conv_0_Relu_ts[6,2]
   + 0.177293062210083 Conv_0_Relu_ts[6,3]
   + 1.534894466400146 Conv_0_Relu_ts[6,4]
   + 0.3099820613861084 Conv_0_Relu_ts[6,5]
   + 1.335353136062622 Conv_0_Relu_ts[6,6]
   - 0.3965733051300049 Conv_0_Relu_ts[6,7]
   + 0.0226722583174706 Conv_0_Relu_ts[6,8]
   - 0.0153551828116179 Conv_0_Relu_ts[6,9]
   - 0.188335582613945 Conv_0_Relu_ts[6,10]
   + 0.1465377509593964 Conv_0_Relu_ts[6,11]
   - 0.4378408491611481 Conv_0_Relu_ts[6,12]
   + 0.5950998067855835 Conv_0_Relu_ts[6,13]
   - 0.2773410975933075 Conv_0_Relu_ts[6,14]
   - 0.3855878710746765 Conv_0_Relu_ts[6,15]
   + 0.0145458607003093 Conv_1_aggregated_features[6,0]
   - 1.349490284919739 Conv_1_aggregated_features[6,1]
   - 0.6738612055778503 Conv_1_aggregated_features[6,2]
   - 0.1977581083774567 Conv_1_aggregated_features[6,3]
   + 3.616869688034058 Conv_1_aggregated_features[6,4]
   - 0.5846084952354431 Conv_1_aggregated_features[6,5]
   + 0.3911246955394745 Conv_1_aggregated_features[6,6]
   - 0.2603789269924164 Conv_1_aggregated_features[6,7]
   + 0.0530429184436798 Conv_1_aggregated_features[6,8]
   + 0.7039647698402405 Conv_1_aggregated_features[6,9]
   - 0.4948081970214844 Conv_1_aggregated_features[6,10]
   - 1.33747124671936 Conv_1_aggregated_features[6,11]
   + 0.4521820247173309 Conv_1_aggregated_features[6,12]
   + 2.879487276077271 Conv_1_aggregated_features[6,13]
   + 0.6494181156158447 Conv_1_aggregated_features[6,14]
   - 0.0441654324531555 Conv_1_aggregated_features[6,15] + Conv_1_t[6,6]
   = -1.141431212425232
 Conv_1_output_constraint[6,7]: 0.1232976838946342 Conv_0_Relu_ts[6,0]
   + 0.7059634327888489 Conv_0_Relu_ts[6,1]
   + 0.0986589044332504 Conv_0_Relu_ts[6,2]
   - 0.1159188747406006 Conv_0_Relu_ts[6,3]
   + 0.1207201555371284 Conv_0_Relu_ts[6,4]
   + 0.4950260519981384 Conv_0_Relu_ts[6,5]
   - 0.1985135078430176 Conv_0_Relu_ts[6,6]
   + 0.3846564292907715 Conv_0_Relu_ts[6,7]
   - 0.2091868817806244 Conv_0_Relu_ts[6,8]
   + 0.0282646510750055 Conv_0_Relu_ts[6,9]
   + 0.00616153934970498 Conv_0_Relu_ts[6,10]
   - 0.2992357313632965 Conv_0_Relu_ts[6,11]
   + 0.6717308759689331 Conv_0_Relu_ts[6,12]
   + 0.3884595632553101 Conv_0_Relu_ts[6,13]
   + 0.1088093742728233 Conv_0_Relu_ts[6,14]
   + 0.1363743990659714 Conv_0_Relu_ts[6,15]
   + 0.2181874513626099 Conv_1_aggregated_features[6,0]
   + 0.1884859055280685 Conv_1_aggregated_features[6,1]
   + 0.1986039727926254 Conv_1_aggregated_features[6,2]
   - 0.0569350719451904 Conv_1_aggregated_features[6,3]
   + 0.4957895576953888 Conv_1_aggregated_features[6,4]
   - 0.2215360552072525 Conv_1_aggregated_features[6,5]
   - 0.3126825392246246 Conv_1_aggregated_features[6,6]
   + 0.2129529714584351 Conv_1_aggregated_features[6,7]
   + 0.1905750632286072 Conv_1_aggregated_features[6,8]
   + 0.3835417926311493 Conv_1_aggregated_features[6,9]
   + 0.0515264272689819 Conv_1_aggregated_features[6,10]
   + 0.2774861454963684 Conv_1_aggregated_features[6,11]
   + 0.1720549166202545 Conv_1_aggregated_features[6,12]
   + 0.2042496353387833 Conv_1_aggregated_features[6,13]
   + 0.1717791706323624 Conv_1_aggregated_features[6,14]
   - 0.0635386854410172 Conv_1_aggregated_features[6,15] + Conv_1_t[6,7]
   = -0.1961010098457336
 Conv_1_output_constraint[6,8]: - 0.0984712317585945 Conv_0_Relu_ts[6,0]
   - 0.8972246050834656 Conv_0_Relu_ts[6,1]
   - 0.1342054009437561 Conv_0_Relu_ts[6,2]
   + 0.0934775173664093 Conv_0_Relu_ts[6,3]
   + 3.929172039031982 Conv_0_Relu_ts[6,4]
   - 1.699812531471252 Conv_0_Relu_ts[6,5]
   + 0.4780299663543701 Conv_0_Relu_ts[6,6]
   + 0.1916014850139618 Conv_0_Relu_ts[6,7]
   - 0.0282735247164965 Conv_0_Relu_ts[6,8]
   - 0.10334312915802 Conv_0_Relu_ts[6,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[6,10]
   - 0.9000989198684692 Conv_0_Relu_ts[6,11]
   - 0.3381493389606476 Conv_0_Relu_ts[6,12]
   + 0.2327869236469269 Conv_0_Relu_ts[6,13]
   + 0.6162272691726685 Conv_0_Relu_ts[6,14]
   - 0.1045716181397438 Conv_0_Relu_ts[6,15]
   - 0.0895406752824783 Conv_1_aggregated_features[6,0]
   - 0.0733596011996269 Conv_1_aggregated_features[6,1]
   - 0.6064443588256836 Conv_1_aggregated_features[6,2]
   + 0.1685465276241302 Conv_1_aggregated_features[6,3]
   + 2.521227836608887 Conv_1_aggregated_features[6,4]
   + 0.8868938088417053 Conv_1_aggregated_features[6,5]
   - 0.3730944097042084 Conv_1_aggregated_features[6,6]
   + 0.4433300495147705 Conv_1_aggregated_features[6,7]
   + 0.2465403079986572 Conv_1_aggregated_features[6,8]
   + 0.1199371814727783 Conv_1_aggregated_features[6,9]
   - 0.427333652973175 Conv_1_aggregated_features[6,10]
   - 1.286244511604309 Conv_1_aggregated_features[6,11]
   + 0.4271916449069977 Conv_1_aggregated_features[6,12]
   + 1.475327014923096 Conv_1_aggregated_features[6,13]
   - 0.2351342439651489 Conv_1_aggregated_features[6,14]
   - 0.1273849010467529 Conv_1_aggregated_features[6,15] + Conv_1_t[6,8]
   = -1.303357601165771
 Conv_1_output_constraint[6,9]: 0.1534582376480103 Conv_0_Relu_ts[6,0]
   + 1.456407189369202 Conv_0_Relu_ts[6,1]
   - 0.0228729248046875 Conv_0_Relu_ts[6,2]
   + 0.1227743625640869 Conv_0_Relu_ts[6,3]
   + 0.7793847918510437 Conv_0_Relu_ts[6,4]
   - 0.2757052779197693 Conv_0_Relu_ts[6,5]
   + 0.6606881022453308 Conv_0_Relu_ts[6,6]
   - 1.107266664505005 Conv_0_Relu_ts[6,7]
   + 0.1723826229572296 Conv_0_Relu_ts[6,8]
   - 0.5540673732757568 Conv_0_Relu_ts[6,9]
   + 0.3133971095085144 Conv_0_Relu_ts[6,10]
   + 0.9318023324012756 Conv_0_Relu_ts[6,11]
   + 0.5855908989906311 Conv_0_Relu_ts[6,12]
   - 0.2116718590259552 Conv_0_Relu_ts[6,13]
   - 0.3693177700042725 Conv_0_Relu_ts[6,14]
   - 0.0642165616154671 Conv_0_Relu_ts[6,15]
   + 0.1819062232971191 Conv_1_aggregated_features[6,0]
   + 1.311912417411804 Conv_1_aggregated_features[6,1]
   + 0.3460695445537567 Conv_1_aggregated_features[6,2]
   + 0.1086616516113281 Conv_1_aggregated_features[6,3]
   - 0.6099227070808411 Conv_1_aggregated_features[6,4]
   + 0.3786990940570831 Conv_1_aggregated_features[6,5]
   + 0.9473716020584106 Conv_1_aggregated_features[6,6]
   + 0.2196423262357712 Conv_1_aggregated_features[6,7]
   + 0.0830722749233246 Conv_1_aggregated_features[6,8]
   - 0.4822347462177277 Conv_1_aggregated_features[6,9]
   + 0.5108910202980042 Conv_1_aggregated_features[6,10]
   + 3.214276790618896 Conv_1_aggregated_features[6,11]
   - 0.7224436402320862 Conv_1_aggregated_features[6,12]
   + 0.2678262591362 Conv_1_aggregated_features[6,13]
   - 0.8807565569877625 Conv_1_aggregated_features[6,14]
   - 0.3268090188503265 Conv_1_aggregated_features[6,15] + Conv_1_t[6,9]
   = -0.8283404111862183
 Conv_1_output_constraint[6,10]: 0.3653882741928101 Conv_0_Relu_ts[6,0]
   - 0.8208209872245789 Conv_0_Relu_ts[6,1]
   + 0.3088222146034241 Conv_0_Relu_ts[6,2]
   + 0.1173163652420044 Conv_0_Relu_ts[6,3]
   + 0.658578634262085 Conv_0_Relu_ts[6,4]
   + 0.584019660949707 Conv_0_Relu_ts[6,5]
   - 0.8654046058654785 Conv_0_Relu_ts[6,6]
   + 0.3697350919246674 Conv_0_Relu_ts[6,7]
   - 0.2274059355258942 Conv_0_Relu_ts[6,8]
   + 1.197938442230225 Conv_0_Relu_ts[6,9]
   + 0.2446341812610626 Conv_0_Relu_ts[6,10]
   - 0.5676743984222412 Conv_0_Relu_ts[6,11]
   - 0.3123646676540375 Conv_0_Relu_ts[6,12]
   + 3.181503534317017 Conv_0_Relu_ts[6,13]
   + 0.6926240921020508 Conv_0_Relu_ts[6,14]
   + 0.0886927545070648 Conv_0_Relu_ts[6,15]
   - 0.2005636394023895 Conv_1_aggregated_features[6,0]
   + 1.403087973594666 Conv_1_aggregated_features[6,1]
   + 0.0523448958992958 Conv_1_aggregated_features[6,2]
   + 0.2411543428897858 Conv_1_aggregated_features[6,3]
   + 0.5953171253204346 Conv_1_aggregated_features[6,4]
   - 0.8278964757919312 Conv_1_aggregated_features[6,5]
   + 0.2499217540025711 Conv_1_aggregated_features[6,6]
   - 0.7393644452095032 Conv_1_aggregated_features[6,7]
   + 0.0902875959873199 Conv_1_aggregated_features[6,8]
   + 0.1488711386919022 Conv_1_aggregated_features[6,9]
   - 0.0303693022578955 Conv_1_aggregated_features[6,10]
   + 2.024244070053101 Conv_1_aggregated_features[6,11]
   + 0.495431512594223 Conv_1_aggregated_features[6,12]
   + 0.3463623225688934 Conv_1_aggregated_features[6,13]
   - 0.0763808861374855 Conv_1_aggregated_features[6,14]
   + 0.1597562879323959 Conv_1_aggregated_features[6,15] + Conv_1_t[6,10]
   = -0.5063248872756958
 Conv_1_output_constraint[6,11]: - 0.5185666680335999 Conv_0_Relu_ts[6,0]
   + 3.656918525695801 Conv_0_Relu_ts[6,1]
   - 0.7082476615905762 Conv_0_Relu_ts[6,2]
   + 0.1922211349010468 Conv_0_Relu_ts[6,3]
   - 0.7149283289909363 Conv_0_Relu_ts[6,4]
   - 3.255658626556396 Conv_0_Relu_ts[6,5]
   + 2.464392185211182 Conv_0_Relu_ts[6,6]
   - 0.9477378129959106 Conv_0_Relu_ts[6,7]
   + 0.168237030506134 Conv_0_Relu_ts[6,8]
   - 1.177640557289124 Conv_0_Relu_ts[6,9]
   - 0.7062549591064453 Conv_0_Relu_ts[6,10]
   + 1.453806400299072 Conv_0_Relu_ts[6,11]
   + 0.2612711191177368 Conv_0_Relu_ts[6,12]
   + 0.2999382615089417 Conv_0_Relu_ts[6,13]
   - 1.255579590797424 Conv_0_Relu_ts[6,14]
   - 0.4265820384025574 Conv_0_Relu_ts[6,15]
   + 0.3379200994968414 Conv_1_aggregated_features[6,0]
   - 1.33229672908783 Conv_1_aggregated_features[6,1]
   - 0.1594242304563522 Conv_1_aggregated_features[6,2]
   + 0.2298935651779175 Conv_1_aggregated_features[6,3]
   - 1.311494588851929 Conv_1_aggregated_features[6,4]
   + 3.605161666870117 Conv_1_aggregated_features[6,5]
   - 0.3176752626895905 Conv_1_aggregated_features[6,6]
   + 0.3926282525062561 Conv_1_aggregated_features[6,7]
   + 0.1479305028915405 Conv_1_aggregated_features[6,8]
   - 0.0372937023639679 Conv_1_aggregated_features[6,9]
   + 0.3455427587032318 Conv_1_aggregated_features[6,10]
   - 2.002299070358276 Conv_1_aggregated_features[6,11]
   - 0.9904713034629822 Conv_1_aggregated_features[6,12]
   + 0.0751789212226868 Conv_1_aggregated_features[6,13]
   - 0.5166233777999878 Conv_1_aggregated_features[6,14]
   + 0.0510688796639442 Conv_1_aggregated_features[6,15] + Conv_1_t[6,11]
   = -1.480785727500916
 Conv_1_output_constraint[6,12]: 0.3080185949802399 Conv_0_Relu_ts[6,0]
   + 0.0994751453399658 Conv_0_Relu_ts[6,1]
   + 0.4488444924354553 Conv_0_Relu_ts[6,2]
   + 0.0305773317813873 Conv_0_Relu_ts[6,3]
   - 1.673393130302429 Conv_0_Relu_ts[6,4]
   + 0.2824500501155853 Conv_0_Relu_ts[6,5]
   - 1.192709684371948 Conv_0_Relu_ts[6,6]
   + 1.454243063926697 Conv_0_Relu_ts[6,7]
   + 0.1514801830053329 Conv_0_Relu_ts[6,8]
   + 1.026351928710938 Conv_0_Relu_ts[6,9]
   + 0.1183762699365616 Conv_0_Relu_ts[6,10]
   + 0.352224588394165 Conv_0_Relu_ts[6,11]
   + 1.090276837348938 Conv_0_Relu_ts[6,12]
   - 2.053535461425781 Conv_0_Relu_ts[6,13]
   + 2.538641452789307 Conv_0_Relu_ts[6,14]
   + 0.6387560963630676 Conv_0_Relu_ts[6,15]
   - 0.1990496516227722 Conv_1_aggregated_features[6,0]
   + 1.247365713119507 Conv_1_aggregated_features[6,1]
   + 0.2213435918092728 Conv_1_aggregated_features[6,2]
   + 0.2398849725723267 Conv_1_aggregated_features[6,3]
   - 2.001479864120483 Conv_1_aggregated_features[6,4]
   - 1.084800243377686 Conv_1_aggregated_features[6,5]
   + 0.4287458956241608 Conv_1_aggregated_features[6,6]
   - 0.8179780840873718 Conv_1_aggregated_features[6,7]
   - 0.0883614718914032 Conv_1_aggregated_features[6,8]
   - 0.4398066997528076 Conv_1_aggregated_features[6,9]
   + 0.2760448455810547 Conv_1_aggregated_features[6,10]
   + 1.944519639015198 Conv_1_aggregated_features[6,11]
   + 1.041437149047852 Conv_1_aggregated_features[6,12]
   - 1.682050704956055 Conv_1_aggregated_features[6,13]
   - 0.1825132220983505 Conv_1_aggregated_features[6,14]
   + 0.2881160080432892 Conv_1_aggregated_features[6,15] + Conv_1_t[6,12]
   = 2.457676649093628
 Conv_1_output_constraint[6,13]: 0.2853958606719971 Conv_0_Relu_ts[6,0]
   + 0.0445350483059883 Conv_0_Relu_ts[6,1]
   - 0.0597315393388271 Conv_0_Relu_ts[6,2]
   + 0.2403226792812347 Conv_0_Relu_ts[6,3]
   + 0.5103238224983215 Conv_0_Relu_ts[6,4]
   - 0.1369927227497101 Conv_0_Relu_ts[6,5]
   + 0.5091087818145752 Conv_0_Relu_ts[6,6]
   - 0.0576926171779633 Conv_0_Relu_ts[6,7]
   - 0.00701320171356201 Conv_0_Relu_ts[6,8]
   + 0.5782948136329651 Conv_0_Relu_ts[6,9]
   - 0.1188536584377289 Conv_0_Relu_ts[6,10]
   - 0.078026682138443 Conv_0_Relu_ts[6,11]
   + 0.1965750753879547 Conv_0_Relu_ts[6,12]
   + 0.4982857704162598 Conv_0_Relu_ts[6,13]
   - 0.0995257869362831 Conv_0_Relu_ts[6,14]
   + 0.0712818428874016 Conv_0_Relu_ts[6,15]
   + 0.2628078460693359 Conv_1_aggregated_features[6,0]
   + 0.6577805280685425 Conv_1_aggregated_features[6,1]
   - 0.1415485441684723 Conv_1_aggregated_features[6,2]
   + 0.1597791016101837 Conv_1_aggregated_features[6,3]
   + 0.3969745337963104 Conv_1_aggregated_features[6,4]
   - 0.3941676616668701 Conv_1_aggregated_features[6,5]
   + 0.3484204709529877 Conv_1_aggregated_features[6,6]
   + 0.2601790428161621 Conv_1_aggregated_features[6,7]
   - 0.0652478933334351 Conv_1_aggregated_features[6,8]
   + 0.3492280542850494 Conv_1_aggregated_features[6,9]
   + 0.1917867064476013 Conv_1_aggregated_features[6,10]
   + 0.2439181953668594 Conv_1_aggregated_features[6,11]
   + 0.1009163707494736 Conv_1_aggregated_features[6,12]
   + 0.2350947707891464 Conv_1_aggregated_features[6,13]
   + 0.2627962231636047 Conv_1_aggregated_features[6,14]
   - 0.0959920063614845 Conv_1_aggregated_features[6,15] + Conv_1_t[6,13]
   = -0.4268372058868408
 Conv_1_output_constraint[6,14]: - 0.2448655962944031 Conv_0_Relu_ts[6,0]
   + 0.2688764035701752 Conv_0_Relu_ts[6,1]
   + 0.0118266940116882 Conv_0_Relu_ts[6,2]
   - 0.0670525431632996 Conv_0_Relu_ts[6,3]
   + 0.1511579602956772 Conv_0_Relu_ts[6,4]
   - 0.4796198606491089 Conv_0_Relu_ts[6,5]
   - 0.1699391901493073 Conv_0_Relu_ts[6,6]
   + 0.2647148370742798 Conv_0_Relu_ts[6,7]
   + 0.137117475271225 Conv_0_Relu_ts[6,8]
   + 0.1737731248140335 Conv_0_Relu_ts[6,9]
   - 0.1610838174819946 Conv_0_Relu_ts[6,10]
   - 0.6311220526695251 Conv_0_Relu_ts[6,11]
   + 0.6263757348060608 Conv_0_Relu_ts[6,12]
   - 0.4823384284973145 Conv_0_Relu_ts[6,13]
   + 0.2192765176296234 Conv_0_Relu_ts[6,14]
   + 0.1739131808280945 Conv_0_Relu_ts[6,15]
   + 0.0443203188478947 Conv_1_aggregated_features[6,0]
   + 0.7668815851211548 Conv_1_aggregated_features[6,1]
   + 0.248365581035614 Conv_1_aggregated_features[6,2]
   - 0.2433404624462128 Conv_1_aggregated_features[6,3]
   + 0.2770798802375793 Conv_1_aggregated_features[6,4]
   - 1.224791765213013 Conv_1_aggregated_features[6,5]
   - 0.1892879456281662 Conv_1_aggregated_features[6,6]
   + 0.1141589805483818 Conv_1_aggregated_features[6,7]
   + 0.2022489905357361 Conv_1_aggregated_features[6,8]
   - 0.0391396470367908 Conv_1_aggregated_features[6,9]
   - 0.0935317650437355 Conv_1_aggregated_features[6,10]
   - 0.5407891273498535 Conv_1_aggregated_features[6,11]
   + 0.3130630552768707 Conv_1_aggregated_features[6,12]
   + 0.3410882651805878 Conv_1_aggregated_features[6,13]
   - 1.855417490005493 Conv_1_aggregated_features[6,14]
   + 0.247613787651062 Conv_1_aggregated_features[6,15] + Conv_1_t[6,14]
   = -0.2306040376424789
 Conv_1_output_constraint[6,15]: 0.043729841709137 Conv_0_Relu_ts[6,0]
   + 0.045797374099493 Conv_0_Relu_ts[6,1]
   - 0.1762047111988068 Conv_0_Relu_ts[6,2]
   - 0.2225021123886108 Conv_0_Relu_ts[6,3]
   + 0.1344186365604401 Conv_0_Relu_ts[6,4]
   + 0.2975046932697296 Conv_0_Relu_ts[6,5]
   + 0.1135610193014145 Conv_0_Relu_ts[6,6]
   + 0.165085032582283 Conv_0_Relu_ts[6,7]
   + 0.0632660016417503 Conv_0_Relu_ts[6,8]
   + 0.2534148395061493 Conv_0_Relu_ts[6,9]
   + 0.1100017726421356 Conv_0_Relu_ts[6,10]
   + 0.045901358127594 Conv_0_Relu_ts[6,11]
   + 0.2102706432342529 Conv_0_Relu_ts[6,12]
   + 0.1767551600933075 Conv_0_Relu_ts[6,13]
   - 0.0864512398838997 Conv_0_Relu_ts[6,14]
   + 0.1616824418306351 Conv_0_Relu_ts[6,15]
   + 0.0113421082496643 Conv_1_aggregated_features[6,0]
   + 0.2387615889310837 Conv_1_aggregated_features[6,1]
   + 0.1763307154178619 Conv_1_aggregated_features[6,2]
   + 0.1442219316959381 Conv_1_aggregated_features[6,3]
   + 0.3573774099349976 Conv_1_aggregated_features[6,4]
   - 0.0406427308917046 Conv_1_aggregated_features[6,5]
   + 0.1155626475811005 Conv_1_aggregated_features[6,6]
   + 0.0186243671923876 Conv_1_aggregated_features[6,7]
   - 0.1389756500720978 Conv_1_aggregated_features[6,8]
   + 0.274962306022644 Conv_1_aggregated_features[6,9]
   + 0.0309249386191368 Conv_1_aggregated_features[6,10]
   - 0.1606657803058624 Conv_1_aggregated_features[6,11]
   + 0.2456246316432953 Conv_1_aggregated_features[6,12]
   + 0.4163841009140015 Conv_1_aggregated_features[6,13]
   + 0.29454505443573 Conv_1_aggregated_features[6,14]
   + 0.1688967496156693 Conv_1_aggregated_features[6,15] + Conv_1_t[6,15]
   = -0.112743653357029
 Conv_1_output_constraint[7,0]: - 0.3061666786670685 Conv_0_Relu_ts[7,0]
   + 0.8854666948318481 Conv_0_Relu_ts[7,1]
   - 1.108363628387451 Conv_0_Relu_ts[7,2]
   - 0.0485447347164154 Conv_0_Relu_ts[7,3]
   - 0.3054571747779846 Conv_0_Relu_ts[7,4]
   - 1.666001558303833 Conv_0_Relu_ts[7,5]
   + 1.575538277626038 Conv_0_Relu_ts[7,6]
   - 0.8786032199859619 Conv_0_Relu_ts[7,7]
   + 0.1696695387363434 Conv_0_Relu_ts[7,8]
   - 0.2058112174272537 Conv_0_Relu_ts[7,9]
   - 0.8619717359542847 Conv_0_Relu_ts[7,10]
   + 1.268710613250732 Conv_0_Relu_ts[7,11]
   + 0.1005195453763008 Conv_0_Relu_ts[7,12]
   - 0.00627355137839913 Conv_0_Relu_ts[7,13]
   - 0.6727985143661499 Conv_0_Relu_ts[7,14]
   - 0.6082888841629028 Conv_0_Relu_ts[7,15]
   + 0.0746944323182106 Conv_1_aggregated_features[7,0]
   - 0.5317568182945251 Conv_1_aggregated_features[7,1]
   + 0.4405037760734558 Conv_1_aggregated_features[7,2]
   - 0.1488203704357147 Conv_1_aggregated_features[7,3]
   + 2.01956033706665 Conv_1_aggregated_features[7,4]
   + 0.8369019031524658 Conv_1_aggregated_features[7,5]
   + 0.1561367213726044 Conv_1_aggregated_features[7,6]
   + 0.1698544919490814 Conv_1_aggregated_features[7,7]
   - 0.0660476386547089 Conv_1_aggregated_features[7,8]
   + 1.570046067237854 Conv_1_aggregated_features[7,9]
   + 0.5540977120399475 Conv_1_aggregated_features[7,10]
   - 1.275410532951355 Conv_1_aggregated_features[7,11]
   - 0.092900924384594 Conv_1_aggregated_features[7,12]
   + 3.287018537521362 Conv_1_aggregated_features[7,13]
   + 2.392175912857056 Conv_1_aggregated_features[7,14]
   + 0.3933185040950775 Conv_1_aggregated_features[7,15] + Conv_1_t[7,0]
   = -2.192610263824463
 Conv_1_output_constraint[7,1]: 0.5202509164810181 Conv_0_Relu_ts[7,0]
   + 0.6686695218086243 Conv_0_Relu_ts[7,1]
   - 0.4720649719238281 Conv_0_Relu_ts[7,2]
   + 0.1026173532009125 Conv_0_Relu_ts[7,3]
   + 0.2593795359134674 Conv_0_Relu_ts[7,4]
   - 2.458898544311523 Conv_0_Relu_ts[7,5]
   - 0.5632196664810181 Conv_0_Relu_ts[7,6]
   - 1.791601657867432 Conv_0_Relu_ts[7,7]
   - 0.1268838942050934 Conv_0_Relu_ts[7,8]
   + 0.7120329737663269 Conv_0_Relu_ts[7,9]
   - 0.5212016701698303 Conv_0_Relu_ts[7,10]
   + 0.6136131286621094 Conv_0_Relu_ts[7,11]
   + 1.563590168952942 Conv_0_Relu_ts[7,12]
   - 0.3512922525405884 Conv_0_Relu_ts[7,13]
   + 0.5421575903892517 Conv_0_Relu_ts[7,14]
   + 0.7331249117851257 Conv_0_Relu_ts[7,15]
   + 0.0113700022920966 Conv_1_aggregated_features[7,0]
   + 2.169116258621216 Conv_1_aggregated_features[7,1]
   + 0.00623909616842866 Conv_1_aggregated_features[7,2]
   + 0.0937856733798981 Conv_1_aggregated_features[7,3]
   + 0.8328568339347839 Conv_1_aggregated_features[7,4]
   + 0.383074939250946 Conv_1_aggregated_features[7,5]
   - 0.1267539262771606 Conv_1_aggregated_features[7,6]
   + 0.6484018564224243 Conv_1_aggregated_features[7,7]
   + 0.1870926916599274 Conv_1_aggregated_features[7,8]
   - 0.4116691052913666 Conv_1_aggregated_features[7,9]
   + 0.1439376026391983 Conv_1_aggregated_features[7,10]
   + 1.05384635925293 Conv_1_aggregated_features[7,11]
   + 1.155439853668213 Conv_1_aggregated_features[7,12]
   - 0.2060132175683975 Conv_1_aggregated_features[7,13]
   + 0.6003169417381287 Conv_1_aggregated_features[7,14]
   + 0.1458597332239151 Conv_1_aggregated_features[7,15] + Conv_1_t[7,1]
   = -0.1077260375022888
 Conv_1_output_constraint[7,2]: 0.4269110858440399 Conv_0_Relu_ts[7,0]
   - 1.097789287567139 Conv_0_Relu_ts[7,1]
   - 0.3380239307880402 Conv_0_Relu_ts[7,2]
   + 0.23756343126297 Conv_0_Relu_ts[7,3]
   - 0.337738424539566 Conv_0_Relu_ts[7,4]
   - 1.407688736915588 Conv_0_Relu_ts[7,5]
   - 0.7854894995689392 Conv_0_Relu_ts[7,6]
   - 1.080899715423584 Conv_0_Relu_ts[7,7]
   + 0.0233876463025808 Conv_0_Relu_ts[7,8]
   + 1.849825739860535 Conv_0_Relu_ts[7,9]
   + 0.4580322206020355 Conv_0_Relu_ts[7,10]
   - 0.819361686706543 Conv_0_Relu_ts[7,11]
   - 0.0822571739554405 Conv_0_Relu_ts[7,12]
   - 0.1501271426677704 Conv_0_Relu_ts[7,13]
   + 0.1831801384687424 Conv_0_Relu_ts[7,14]
   + 0.00308510870672762 Conv_0_Relu_ts[7,15]
   - 0.2153502702713013 Conv_1_aggregated_features[7,0]
   + 3.817557334899902 Conv_1_aggregated_features[7,1]
   - 0.3402683734893799 Conv_1_aggregated_features[7,2]
   + 0.0650523602962494 Conv_1_aggregated_features[7,3]
   - 0.4435569643974304 Conv_1_aggregated_features[7,4]
   + 0.045248981565237 Conv_1_aggregated_features[7,5]
   + 0.844337522983551 Conv_1_aggregated_features[7,6]
   - 0.0691490024328232 Conv_1_aggregated_features[7,7]
   + 0.0830505788326263 Conv_1_aggregated_features[7,8]
   - 0.3022302687168121 Conv_1_aggregated_features[7,9]
   - 0.611566960811615 Conv_1_aggregated_features[7,10]
   + 3.582128524780273 Conv_1_aggregated_features[7,11]
   + 0.1314867585897446 Conv_1_aggregated_features[7,12]
   + 0.5333523154258728 Conv_1_aggregated_features[7,13]
   - 0.1983102858066559 Conv_1_aggregated_features[7,14]
   + 0.1628565639257431 Conv_1_aggregated_features[7,15] + Conv_1_t[7,2]
   = -0.4443503022193909
 Conv_1_output_constraint[7,3]: 0.066770151257515 Conv_0_Relu_ts[7,0]
   + 0.3136637210845947 Conv_0_Relu_ts[7,1]
   + 0.2078424692153931 Conv_0_Relu_ts[7,2]
   - 0.1673517823219299 Conv_0_Relu_ts[7,3]
   + 1.022224068641663 Conv_0_Relu_ts[7,4]
   - 0.1723377108573914 Conv_0_Relu_ts[7,5]
   + 0.3032898306846619 Conv_0_Relu_ts[7,6]
   + 0.0841449126601219 Conv_0_Relu_ts[7,7]
   + 0.0921086072921753 Conv_0_Relu_ts[7,8]
   - 0.022469462826848 Conv_0_Relu_ts[7,9]
   - 0.1657154262065887 Conv_0_Relu_ts[7,10]
   + 1.007031679153442 Conv_0_Relu_ts[7,11]
   + 0.0208180602639914 Conv_0_Relu_ts[7,12]
   + 0.3933880031108856 Conv_0_Relu_ts[7,13]
   + 0.3077191412448883 Conv_0_Relu_ts[7,14]
   + 0.0400979965925217 Conv_0_Relu_ts[7,15]
   + 0.0230858232825994 Conv_1_aggregated_features[7,0]
   + 0.3792195022106171 Conv_1_aggregated_features[7,1]
   + 0.1572305560112 Conv_1_aggregated_features[7,2]
   - 0.0165807008743286 Conv_1_aggregated_features[7,3]
   + 0.5649795532226562 Conv_1_aggregated_features[7,4]
   + 0.6253551840782166 Conv_1_aggregated_features[7,5]
   + 0.3968201577663422 Conv_1_aggregated_features[7,6]
   + 0.2728661298751831 Conv_1_aggregated_features[7,7]
   - 0.210608571767807 Conv_1_aggregated_features[7,8]
   + 0.094737209379673 Conv_1_aggregated_features[7,9]
   - 0.1354089975357056 Conv_1_aggregated_features[7,10]
   + 0.9330839514732361 Conv_1_aggregated_features[7,11]
   + 0.1677136421203613 Conv_1_aggregated_features[7,12]
   + 0.6573846936225891 Conv_1_aggregated_features[7,13]
   + 0.044663343578577 Conv_1_aggregated_features[7,14]
   - 0.0132347755134106 Conv_1_aggregated_features[7,15] + Conv_1_t[7,3]
   = -0.327608048915863
 Conv_1_output_constraint[7,4]: - 0.0720538049936295 Conv_0_Relu_ts[7,0]
   + 3.730369091033936 Conv_0_Relu_ts[7,1]
   - 0.8118388652801514 Conv_0_Relu_ts[7,2]
   - 0.174347460269928 Conv_0_Relu_ts[7,3]
   - 3.084089279174805 Conv_0_Relu_ts[7,4]
   - 5.146248340606689 Conv_0_Relu_ts[7,5]
   - 2.440467357635498 Conv_0_Relu_ts[7,6]
   - 3.765322923660278 Conv_0_Relu_ts[7,7]
   + 0.153628796339035 Conv_0_Relu_ts[7,8]
   + 0.1013353765010834 Conv_0_Relu_ts[7,9]
   - 0.5803871154785156 Conv_0_Relu_ts[7,10]
   - 2.657181739807129 Conv_0_Relu_ts[7,11]
   + 0.6840037703514099 Conv_0_Relu_ts[7,12]
   - 1.184461355209351 Conv_0_Relu_ts[7,13]
   - 0.3772841989994049 Conv_0_Relu_ts[7,14]
   + 0.5856359601020813 Conv_0_Relu_ts[7,15]
   + 0.2062403708696365 Conv_1_aggregated_features[7,0]
   + 1.470675468444824 Conv_1_aggregated_features[7,1]
   + 0.3523006737232208 Conv_1_aggregated_features[7,2]
   - 0.1351108849048615 Conv_1_aggregated_features[7,3]
   + 0.8932182788848877 Conv_1_aggregated_features[7,4]
   + 1.194340229034424 Conv_1_aggregated_features[7,5]
   + 0.0163087919354439 Conv_1_aggregated_features[7,6]
   - 0.5082771182060242 Conv_1_aggregated_features[7,7]
   - 0.0201514065265656 Conv_1_aggregated_features[7,8]
   + 0.9045064449310303 Conv_1_aggregated_features[7,9]
   + 0.7243757843971252 Conv_1_aggregated_features[7,10]
   + 0.5823414921760559 Conv_1_aggregated_features[7,11]
   + 0.0738087296485901 Conv_1_aggregated_features[7,12]
   - 0.3080458641052246 Conv_1_aggregated_features[7,13]
   + 0.9457569122314453 Conv_1_aggregated_features[7,14]
   + 0.4423830211162567 Conv_1_aggregated_features[7,15] + Conv_1_t[7,4]
   = -0.2138544023036957
 Conv_1_output_constraint[7,5]: 0.2116706073284149 Conv_0_Relu_ts[7,0]
   + 0.2319677472114563 Conv_0_Relu_ts[7,1]
   + 0.1967353820800781 Conv_0_Relu_ts[7,2]
   - 0.1974936425685883 Conv_0_Relu_ts[7,3]
   + 0.087493434548378 Conv_0_Relu_ts[7,4]
   + 0.0226023308932781 Conv_0_Relu_ts[7,5]
   - 0.1455883532762527 Conv_0_Relu_ts[7,6]
   - 0.1098284423351288 Conv_0_Relu_ts[7,7]
   + 0.0375414304435253 Conv_0_Relu_ts[7,8]
   + 0.2209681868553162 Conv_0_Relu_ts[7,9]
   + 0.0180207490921021 Conv_0_Relu_ts[7,10]
   - 0.0944192707538605 Conv_0_Relu_ts[7,11]
   + 0.1318628787994385 Conv_0_Relu_ts[7,12]
   + 0.1718421876430511 Conv_0_Relu_ts[7,13]
   - 0.0992768108844757 Conv_0_Relu_ts[7,14]
   + 0.2220577001571655 Conv_0_Relu_ts[7,15]
   + 0.0500434041023254 Conv_1_aggregated_features[7,0]
   + 0.2417693138122559 Conv_1_aggregated_features[7,1]
   + 0.1884889900684357 Conv_1_aggregated_features[7,2]
   - 0.0459566116333008 Conv_1_aggregated_features[7,3]
   - 0.1095028519630432 Conv_1_aggregated_features[7,4]
   - 0.1494371891021729 Conv_1_aggregated_features[7,5]
   - 0.0601528584957123 Conv_1_aggregated_features[7,6]
   - 0.2431226968765259 Conv_1_aggregated_features[7,7]
   + 0.0709281861782074 Conv_1_aggregated_features[7,8]
   + 0.1859873235225677 Conv_1_aggregated_features[7,9]
   - 0.0025022029876709 Conv_1_aggregated_features[7,10]
   + 0.1165908575057983 Conv_1_aggregated_features[7,11]
   - 0.0946725904941559 Conv_1_aggregated_features[7,12]
   + 0.1679102778434753 Conv_1_aggregated_features[7,13]
   + 0.0888539850711823 Conv_1_aggregated_features[7,14]
   + 0.0481749773025513 Conv_1_aggregated_features[7,15] + Conv_1_t[7,5]
   = -0.2017861604690552
 Conv_1_output_constraint[7,6]: - 0.5161762237548828 Conv_0_Relu_ts[7,0]
   - 0.9039249420166016 Conv_0_Relu_ts[7,1]
   - 0.5091935992240906 Conv_0_Relu_ts[7,2]
   + 0.177293062210083 Conv_0_Relu_ts[7,3]
   + 1.534894466400146 Conv_0_Relu_ts[7,4]
   + 0.3099820613861084 Conv_0_Relu_ts[7,5]
   + 1.335353136062622 Conv_0_Relu_ts[7,6]
   - 0.3965733051300049 Conv_0_Relu_ts[7,7]
   + 0.0226722583174706 Conv_0_Relu_ts[7,8]
   - 0.0153551828116179 Conv_0_Relu_ts[7,9]
   - 0.188335582613945 Conv_0_Relu_ts[7,10]
   + 0.1465377509593964 Conv_0_Relu_ts[7,11]
   - 0.4378408491611481 Conv_0_Relu_ts[7,12]
   + 0.5950998067855835 Conv_0_Relu_ts[7,13]
   - 0.2773410975933075 Conv_0_Relu_ts[7,14]
   - 0.3855878710746765 Conv_0_Relu_ts[7,15]
   + 0.0145458607003093 Conv_1_aggregated_features[7,0]
   - 1.349490284919739 Conv_1_aggregated_features[7,1]
   - 0.6738612055778503 Conv_1_aggregated_features[7,2]
   - 0.1977581083774567 Conv_1_aggregated_features[7,3]
   + 3.616869688034058 Conv_1_aggregated_features[7,4]
   - 0.5846084952354431 Conv_1_aggregated_features[7,5]
   + 0.3911246955394745 Conv_1_aggregated_features[7,6]
   - 0.2603789269924164 Conv_1_aggregated_features[7,7]
   + 0.0530429184436798 Conv_1_aggregated_features[7,8]
   + 0.7039647698402405 Conv_1_aggregated_features[7,9]
   - 0.4948081970214844 Conv_1_aggregated_features[7,10]
   - 1.33747124671936 Conv_1_aggregated_features[7,11]
   + 0.4521820247173309 Conv_1_aggregated_features[7,12]
   + 2.879487276077271 Conv_1_aggregated_features[7,13]
   + 0.6494181156158447 Conv_1_aggregated_features[7,14]
   - 0.0441654324531555 Conv_1_aggregated_features[7,15] + Conv_1_t[7,6]
   = -1.141431212425232
 Conv_1_output_constraint[7,7]: 0.1232976838946342 Conv_0_Relu_ts[7,0]
   + 0.7059634327888489 Conv_0_Relu_ts[7,1]
   + 0.0986589044332504 Conv_0_Relu_ts[7,2]
   - 0.1159188747406006 Conv_0_Relu_ts[7,3]
   + 0.1207201555371284 Conv_0_Relu_ts[7,4]
   + 0.4950260519981384 Conv_0_Relu_ts[7,5]
   - 0.1985135078430176 Conv_0_Relu_ts[7,6]
   + 0.3846564292907715 Conv_0_Relu_ts[7,7]
   - 0.2091868817806244 Conv_0_Relu_ts[7,8]
   + 0.0282646510750055 Conv_0_Relu_ts[7,9]
   + 0.00616153934970498 Conv_0_Relu_ts[7,10]
   - 0.2992357313632965 Conv_0_Relu_ts[7,11]
   + 0.6717308759689331 Conv_0_Relu_ts[7,12]
   + 0.3884595632553101 Conv_0_Relu_ts[7,13]
   + 0.1088093742728233 Conv_0_Relu_ts[7,14]
   + 0.1363743990659714 Conv_0_Relu_ts[7,15]
   + 0.2181874513626099 Conv_1_aggregated_features[7,0]
   + 0.1884859055280685 Conv_1_aggregated_features[7,1]
   + 0.1986039727926254 Conv_1_aggregated_features[7,2]
   - 0.0569350719451904 Conv_1_aggregated_features[7,3]
   + 0.4957895576953888 Conv_1_aggregated_features[7,4]
   - 0.2215360552072525 Conv_1_aggregated_features[7,5]
   - 0.3126825392246246 Conv_1_aggregated_features[7,6]
   + 0.2129529714584351 Conv_1_aggregated_features[7,7]
   + 0.1905750632286072 Conv_1_aggregated_features[7,8]
   + 0.3835417926311493 Conv_1_aggregated_features[7,9]
   + 0.0515264272689819 Conv_1_aggregated_features[7,10]
   + 0.2774861454963684 Conv_1_aggregated_features[7,11]
   + 0.1720549166202545 Conv_1_aggregated_features[7,12]
   + 0.2042496353387833 Conv_1_aggregated_features[7,13]
   + 0.1717791706323624 Conv_1_aggregated_features[7,14]
   - 0.0635386854410172 Conv_1_aggregated_features[7,15] + Conv_1_t[7,7]
   = -0.1961010098457336
 Conv_1_output_constraint[7,8]: - 0.0984712317585945 Conv_0_Relu_ts[7,0]
   - 0.8972246050834656 Conv_0_Relu_ts[7,1]
   - 0.1342054009437561 Conv_0_Relu_ts[7,2]
   + 0.0934775173664093 Conv_0_Relu_ts[7,3]
   + 3.929172039031982 Conv_0_Relu_ts[7,4]
   - 1.699812531471252 Conv_0_Relu_ts[7,5]
   + 0.4780299663543701 Conv_0_Relu_ts[7,6]
   + 0.1916014850139618 Conv_0_Relu_ts[7,7]
   - 0.0282735247164965 Conv_0_Relu_ts[7,8]
   - 0.10334312915802 Conv_0_Relu_ts[7,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[7,10]
   - 0.9000989198684692 Conv_0_Relu_ts[7,11]
   - 0.3381493389606476 Conv_0_Relu_ts[7,12]
   + 0.2327869236469269 Conv_0_Relu_ts[7,13]
   + 0.6162272691726685 Conv_0_Relu_ts[7,14]
   - 0.1045716181397438 Conv_0_Relu_ts[7,15]
   - 0.0895406752824783 Conv_1_aggregated_features[7,0]
   - 0.0733596011996269 Conv_1_aggregated_features[7,1]
   - 0.6064443588256836 Conv_1_aggregated_features[7,2]
   + 0.1685465276241302 Conv_1_aggregated_features[7,3]
   + 2.521227836608887 Conv_1_aggregated_features[7,4]
   + 0.8868938088417053 Conv_1_aggregated_features[7,5]
   - 0.3730944097042084 Conv_1_aggregated_features[7,6]
   + 0.4433300495147705 Conv_1_aggregated_features[7,7]
   + 0.2465403079986572 Conv_1_aggregated_features[7,8]
   + 0.1199371814727783 Conv_1_aggregated_features[7,9]
   - 0.427333652973175 Conv_1_aggregated_features[7,10]
   - 1.286244511604309 Conv_1_aggregated_features[7,11]
   + 0.4271916449069977 Conv_1_aggregated_features[7,12]
   + 1.475327014923096 Conv_1_aggregated_features[7,13]
   - 0.2351342439651489 Conv_1_aggregated_features[7,14]
   - 0.1273849010467529 Conv_1_aggregated_features[7,15] + Conv_1_t[7,8]
   = -1.303357601165771
 Conv_1_output_constraint[7,9]: 0.1534582376480103 Conv_0_Relu_ts[7,0]
   + 1.456407189369202 Conv_0_Relu_ts[7,1]
   - 0.0228729248046875 Conv_0_Relu_ts[7,2]
   + 0.1227743625640869 Conv_0_Relu_ts[7,3]
   + 0.7793847918510437 Conv_0_Relu_ts[7,4]
   - 0.2757052779197693 Conv_0_Relu_ts[7,5]
   + 0.6606881022453308 Conv_0_Relu_ts[7,6]
   - 1.107266664505005 Conv_0_Relu_ts[7,7]
   + 0.1723826229572296 Conv_0_Relu_ts[7,8]
   - 0.5540673732757568 Conv_0_Relu_ts[7,9]
   + 0.3133971095085144 Conv_0_Relu_ts[7,10]
   + 0.9318023324012756 Conv_0_Relu_ts[7,11]
   + 0.5855908989906311 Conv_0_Relu_ts[7,12]
   - 0.2116718590259552 Conv_0_Relu_ts[7,13]
   - 0.3693177700042725 Conv_0_Relu_ts[7,14]
   - 0.0642165616154671 Conv_0_Relu_ts[7,15]
   + 0.1819062232971191 Conv_1_aggregated_features[7,0]
   + 1.311912417411804 Conv_1_aggregated_features[7,1]
   + 0.3460695445537567 Conv_1_aggregated_features[7,2]
   + 0.1086616516113281 Conv_1_aggregated_features[7,3]
   - 0.6099227070808411 Conv_1_aggregated_features[7,4]
   + 0.3786990940570831 Conv_1_aggregated_features[7,5]
   + 0.9473716020584106 Conv_1_aggregated_features[7,6]
   + 0.2196423262357712 Conv_1_aggregated_features[7,7]
   + 0.0830722749233246 Conv_1_aggregated_features[7,8]
   - 0.4822347462177277 Conv_1_aggregated_features[7,9]
   + 0.5108910202980042 Conv_1_aggregated_features[7,10]
   + 3.214276790618896 Conv_1_aggregated_features[7,11]
   - 0.7224436402320862 Conv_1_aggregated_features[7,12]
   + 0.2678262591362 Conv_1_aggregated_features[7,13]
   - 0.8807565569877625 Conv_1_aggregated_features[7,14]
   - 0.3268090188503265 Conv_1_aggregated_features[7,15] + Conv_1_t[7,9]
   = -0.8283404111862183
 Conv_1_output_constraint[7,10]: 0.3653882741928101 Conv_0_Relu_ts[7,0]
   - 0.8208209872245789 Conv_0_Relu_ts[7,1]
   + 0.3088222146034241 Conv_0_Relu_ts[7,2]
   + 0.1173163652420044 Conv_0_Relu_ts[7,3]
   + 0.658578634262085 Conv_0_Relu_ts[7,4]
   + 0.584019660949707 Conv_0_Relu_ts[7,5]
   - 0.8654046058654785 Conv_0_Relu_ts[7,6]
   + 0.3697350919246674 Conv_0_Relu_ts[7,7]
   - 0.2274059355258942 Conv_0_Relu_ts[7,8]
   + 1.197938442230225 Conv_0_Relu_ts[7,9]
   + 0.2446341812610626 Conv_0_Relu_ts[7,10]
   - 0.5676743984222412 Conv_0_Relu_ts[7,11]
   - 0.3123646676540375 Conv_0_Relu_ts[7,12]
   + 3.181503534317017 Conv_0_Relu_ts[7,13]
   + 0.6926240921020508 Conv_0_Relu_ts[7,14]
   + 0.0886927545070648 Conv_0_Relu_ts[7,15]
   - 0.2005636394023895 Conv_1_aggregated_features[7,0]
   + 1.403087973594666 Conv_1_aggregated_features[7,1]
   + 0.0523448958992958 Conv_1_aggregated_features[7,2]
   + 0.2411543428897858 Conv_1_aggregated_features[7,3]
   + 0.5953171253204346 Conv_1_aggregated_features[7,4]
   - 0.8278964757919312 Conv_1_aggregated_features[7,5]
   + 0.2499217540025711 Conv_1_aggregated_features[7,6]
   - 0.7393644452095032 Conv_1_aggregated_features[7,7]
   + 0.0902875959873199 Conv_1_aggregated_features[7,8]
   + 0.1488711386919022 Conv_1_aggregated_features[7,9]
   - 0.0303693022578955 Conv_1_aggregated_features[7,10]
   + 2.024244070053101 Conv_1_aggregated_features[7,11]
   + 0.495431512594223 Conv_1_aggregated_features[7,12]
   + 0.3463623225688934 Conv_1_aggregated_features[7,13]
   - 0.0763808861374855 Conv_1_aggregated_features[7,14]
   + 0.1597562879323959 Conv_1_aggregated_features[7,15] + Conv_1_t[7,10]
   = -0.5063248872756958
 Conv_1_output_constraint[7,11]: - 0.5185666680335999 Conv_0_Relu_ts[7,0]
   + 3.656918525695801 Conv_0_Relu_ts[7,1]
   - 0.7082476615905762 Conv_0_Relu_ts[7,2]
   + 0.1922211349010468 Conv_0_Relu_ts[7,3]
   - 0.7149283289909363 Conv_0_Relu_ts[7,4]
   - 3.255658626556396 Conv_0_Relu_ts[7,5]
   + 2.464392185211182 Conv_0_Relu_ts[7,6]
   - 0.9477378129959106 Conv_0_Relu_ts[7,7]
   + 0.168237030506134 Conv_0_Relu_ts[7,8]
   - 1.177640557289124 Conv_0_Relu_ts[7,9]
   - 0.7062549591064453 Conv_0_Relu_ts[7,10]
   + 1.453806400299072 Conv_0_Relu_ts[7,11]
   + 0.2612711191177368 Conv_0_Relu_ts[7,12]
   + 0.2999382615089417 Conv_0_Relu_ts[7,13]
   - 1.255579590797424 Conv_0_Relu_ts[7,14]
   - 0.4265820384025574 Conv_0_Relu_ts[7,15]
   + 0.3379200994968414 Conv_1_aggregated_features[7,0]
   - 1.33229672908783 Conv_1_aggregated_features[7,1]
   - 0.1594242304563522 Conv_1_aggregated_features[7,2]
   + 0.2298935651779175 Conv_1_aggregated_features[7,3]
   - 1.311494588851929 Conv_1_aggregated_features[7,4]
   + 3.605161666870117 Conv_1_aggregated_features[7,5]
   - 0.3176752626895905 Conv_1_aggregated_features[7,6]
   + 0.3926282525062561 Conv_1_aggregated_features[7,7]
   + 0.1479305028915405 Conv_1_aggregated_features[7,8]
   - 0.0372937023639679 Conv_1_aggregated_features[7,9]
   + 0.3455427587032318 Conv_1_aggregated_features[7,10]
   - 2.002299070358276 Conv_1_aggregated_features[7,11]
   - 0.9904713034629822 Conv_1_aggregated_features[7,12]
   + 0.0751789212226868 Conv_1_aggregated_features[7,13]
   - 0.5166233777999878 Conv_1_aggregated_features[7,14]
   + 0.0510688796639442 Conv_1_aggregated_features[7,15] + Conv_1_t[7,11]
   = -1.480785727500916
 Conv_1_output_constraint[7,12]: 0.3080185949802399 Conv_0_Relu_ts[7,0]
   + 0.0994751453399658 Conv_0_Relu_ts[7,1]
   + 0.4488444924354553 Conv_0_Relu_ts[7,2]
   + 0.0305773317813873 Conv_0_Relu_ts[7,3]
   - 1.673393130302429 Conv_0_Relu_ts[7,4]
   + 0.2824500501155853 Conv_0_Relu_ts[7,5]
   - 1.192709684371948 Conv_0_Relu_ts[7,6]
   + 1.454243063926697 Conv_0_Relu_ts[7,7]
   + 0.1514801830053329 Conv_0_Relu_ts[7,8]
   + 1.026351928710938 Conv_0_Relu_ts[7,9]
   + 0.1183762699365616 Conv_0_Relu_ts[7,10]
   + 0.352224588394165 Conv_0_Relu_ts[7,11]
   + 1.090276837348938 Conv_0_Relu_ts[7,12]
   - 2.053535461425781 Conv_0_Relu_ts[7,13]
   + 2.538641452789307 Conv_0_Relu_ts[7,14]
   + 0.6387560963630676 Conv_0_Relu_ts[7,15]
   - 0.1990496516227722 Conv_1_aggregated_features[7,0]
   + 1.247365713119507 Conv_1_aggregated_features[7,1]
   + 0.2213435918092728 Conv_1_aggregated_features[7,2]
   + 0.2398849725723267 Conv_1_aggregated_features[7,3]
   - 2.001479864120483 Conv_1_aggregated_features[7,4]
   - 1.084800243377686 Conv_1_aggregated_features[7,5]
   + 0.4287458956241608 Conv_1_aggregated_features[7,6]
   - 0.8179780840873718 Conv_1_aggregated_features[7,7]
   - 0.0883614718914032 Conv_1_aggregated_features[7,8]
   - 0.4398066997528076 Conv_1_aggregated_features[7,9]
   + 0.2760448455810547 Conv_1_aggregated_features[7,10]
   + 1.944519639015198 Conv_1_aggregated_features[7,11]
   + 1.041437149047852 Conv_1_aggregated_features[7,12]
   - 1.682050704956055 Conv_1_aggregated_features[7,13]
   - 0.1825132220983505 Conv_1_aggregated_features[7,14]
   + 0.2881160080432892 Conv_1_aggregated_features[7,15] + Conv_1_t[7,12]
   = 2.457676649093628
 Conv_1_output_constraint[7,13]: 0.2853958606719971 Conv_0_Relu_ts[7,0]
   + 0.0445350483059883 Conv_0_Relu_ts[7,1]
   - 0.0597315393388271 Conv_0_Relu_ts[7,2]
   + 0.2403226792812347 Conv_0_Relu_ts[7,3]
   + 0.5103238224983215 Conv_0_Relu_ts[7,4]
   - 0.1369927227497101 Conv_0_Relu_ts[7,5]
   + 0.5091087818145752 Conv_0_Relu_ts[7,6]
   - 0.0576926171779633 Conv_0_Relu_ts[7,7]
   - 0.00701320171356201 Conv_0_Relu_ts[7,8]
   + 0.5782948136329651 Conv_0_Relu_ts[7,9]
   - 0.1188536584377289 Conv_0_Relu_ts[7,10]
   - 0.078026682138443 Conv_0_Relu_ts[7,11]
   + 0.1965750753879547 Conv_0_Relu_ts[7,12]
   + 0.4982857704162598 Conv_0_Relu_ts[7,13]
   - 0.0995257869362831 Conv_0_Relu_ts[7,14]
   + 0.0712818428874016 Conv_0_Relu_ts[7,15]
   + 0.2628078460693359 Conv_1_aggregated_features[7,0]
   + 0.6577805280685425 Conv_1_aggregated_features[7,1]
   - 0.1415485441684723 Conv_1_aggregated_features[7,2]
   + 0.1597791016101837 Conv_1_aggregated_features[7,3]
   + 0.3969745337963104 Conv_1_aggregated_features[7,4]
   - 0.3941676616668701 Conv_1_aggregated_features[7,5]
   + 0.3484204709529877 Conv_1_aggregated_features[7,6]
   + 0.2601790428161621 Conv_1_aggregated_features[7,7]
   - 0.0652478933334351 Conv_1_aggregated_features[7,8]
   + 0.3492280542850494 Conv_1_aggregated_features[7,9]
   + 0.1917867064476013 Conv_1_aggregated_features[7,10]
   + 0.2439181953668594 Conv_1_aggregated_features[7,11]
   + 0.1009163707494736 Conv_1_aggregated_features[7,12]
   + 0.2350947707891464 Conv_1_aggregated_features[7,13]
   + 0.2627962231636047 Conv_1_aggregated_features[7,14]
   - 0.0959920063614845 Conv_1_aggregated_features[7,15] + Conv_1_t[7,13]
   = -0.4268372058868408
 Conv_1_output_constraint[7,14]: - 0.2448655962944031 Conv_0_Relu_ts[7,0]
   + 0.2688764035701752 Conv_0_Relu_ts[7,1]
   + 0.0118266940116882 Conv_0_Relu_ts[7,2]
   - 0.0670525431632996 Conv_0_Relu_ts[7,3]
   + 0.1511579602956772 Conv_0_Relu_ts[7,4]
   - 0.4796198606491089 Conv_0_Relu_ts[7,5]
   - 0.1699391901493073 Conv_0_Relu_ts[7,6]
   + 0.2647148370742798 Conv_0_Relu_ts[7,7]
   + 0.137117475271225 Conv_0_Relu_ts[7,8]
   + 0.1737731248140335 Conv_0_Relu_ts[7,9]
   - 0.1610838174819946 Conv_0_Relu_ts[7,10]
   - 0.6311220526695251 Conv_0_Relu_ts[7,11]
   + 0.6263757348060608 Conv_0_Relu_ts[7,12]
   - 0.4823384284973145 Conv_0_Relu_ts[7,13]
   + 0.2192765176296234 Conv_0_Relu_ts[7,14]
   + 0.1739131808280945 Conv_0_Relu_ts[7,15]
   + 0.0443203188478947 Conv_1_aggregated_features[7,0]
   + 0.7668815851211548 Conv_1_aggregated_features[7,1]
   + 0.248365581035614 Conv_1_aggregated_features[7,2]
   - 0.2433404624462128 Conv_1_aggregated_features[7,3]
   + 0.2770798802375793 Conv_1_aggregated_features[7,4]
   - 1.224791765213013 Conv_1_aggregated_features[7,5]
   - 0.1892879456281662 Conv_1_aggregated_features[7,6]
   + 0.1141589805483818 Conv_1_aggregated_features[7,7]
   + 0.2022489905357361 Conv_1_aggregated_features[7,8]
   - 0.0391396470367908 Conv_1_aggregated_features[7,9]
   - 0.0935317650437355 Conv_1_aggregated_features[7,10]
   - 0.5407891273498535 Conv_1_aggregated_features[7,11]
   + 0.3130630552768707 Conv_1_aggregated_features[7,12]
   + 0.3410882651805878 Conv_1_aggregated_features[7,13]
   - 1.855417490005493 Conv_1_aggregated_features[7,14]
   + 0.247613787651062 Conv_1_aggregated_features[7,15] + Conv_1_t[7,14]
   = -0.2306040376424789
 Conv_1_output_constraint[7,15]: 0.043729841709137 Conv_0_Relu_ts[7,0]
   + 0.045797374099493 Conv_0_Relu_ts[7,1]
   - 0.1762047111988068 Conv_0_Relu_ts[7,2]
   - 0.2225021123886108 Conv_0_Relu_ts[7,3]
   + 0.1344186365604401 Conv_0_Relu_ts[7,4]
   + 0.2975046932697296 Conv_0_Relu_ts[7,5]
   + 0.1135610193014145 Conv_0_Relu_ts[7,6]
   + 0.165085032582283 Conv_0_Relu_ts[7,7]
   + 0.0632660016417503 Conv_0_Relu_ts[7,8]
   + 0.2534148395061493 Conv_0_Relu_ts[7,9]
   + 0.1100017726421356 Conv_0_Relu_ts[7,10]
   + 0.045901358127594 Conv_0_Relu_ts[7,11]
   + 0.2102706432342529 Conv_0_Relu_ts[7,12]
   + 0.1767551600933075 Conv_0_Relu_ts[7,13]
   - 0.0864512398838997 Conv_0_Relu_ts[7,14]
   + 0.1616824418306351 Conv_0_Relu_ts[7,15]
   + 0.0113421082496643 Conv_1_aggregated_features[7,0]
   + 0.2387615889310837 Conv_1_aggregated_features[7,1]
   + 0.1763307154178619 Conv_1_aggregated_features[7,2]
   + 0.1442219316959381 Conv_1_aggregated_features[7,3]
   + 0.3573774099349976 Conv_1_aggregated_features[7,4]
   - 0.0406427308917046 Conv_1_aggregated_features[7,5]
   + 0.1155626475811005 Conv_1_aggregated_features[7,6]
   + 0.0186243671923876 Conv_1_aggregated_features[7,7]
   - 0.1389756500720978 Conv_1_aggregated_features[7,8]
   + 0.274962306022644 Conv_1_aggregated_features[7,9]
   + 0.0309249386191368 Conv_1_aggregated_features[7,10]
   - 0.1606657803058624 Conv_1_aggregated_features[7,11]
   + 0.2456246316432953 Conv_1_aggregated_features[7,12]
   + 0.4163841009140015 Conv_1_aggregated_features[7,13]
   + 0.29454505443573 Conv_1_aggregated_features[7,14]
   + 0.1688967496156693 Conv_1_aggregated_features[7,15] + Conv_1_t[7,15]
   = -0.112743653357029
 Conv_1_output_constraint[8,0]: - 0.3061666786670685 Conv_0_Relu_ts[8,0]
   + 0.8854666948318481 Conv_0_Relu_ts[8,1]
   - 1.108363628387451 Conv_0_Relu_ts[8,2]
   - 0.0485447347164154 Conv_0_Relu_ts[8,3]
   - 0.3054571747779846 Conv_0_Relu_ts[8,4]
   - 1.666001558303833 Conv_0_Relu_ts[8,5]
   + 1.575538277626038 Conv_0_Relu_ts[8,6]
   - 0.8786032199859619 Conv_0_Relu_ts[8,7]
   + 0.1696695387363434 Conv_0_Relu_ts[8,8]
   - 0.2058112174272537 Conv_0_Relu_ts[8,9]
   - 0.8619717359542847 Conv_0_Relu_ts[8,10]
   + 1.268710613250732 Conv_0_Relu_ts[8,11]
   + 0.1005195453763008 Conv_0_Relu_ts[8,12]
   - 0.00627355137839913 Conv_0_Relu_ts[8,13]
   - 0.6727985143661499 Conv_0_Relu_ts[8,14]
   - 0.6082888841629028 Conv_0_Relu_ts[8,15]
   + 0.0746944323182106 Conv_1_aggregated_features[8,0]
   - 0.5317568182945251 Conv_1_aggregated_features[8,1]
   + 0.4405037760734558 Conv_1_aggregated_features[8,2]
   - 0.1488203704357147 Conv_1_aggregated_features[8,3]
   + 2.01956033706665 Conv_1_aggregated_features[8,4]
   + 0.8369019031524658 Conv_1_aggregated_features[8,5]
   + 0.1561367213726044 Conv_1_aggregated_features[8,6]
   + 0.1698544919490814 Conv_1_aggregated_features[8,7]
   - 0.0660476386547089 Conv_1_aggregated_features[8,8]
   + 1.570046067237854 Conv_1_aggregated_features[8,9]
   + 0.5540977120399475 Conv_1_aggregated_features[8,10]
   - 1.275410532951355 Conv_1_aggregated_features[8,11]
   - 0.092900924384594 Conv_1_aggregated_features[8,12]
   + 3.287018537521362 Conv_1_aggregated_features[8,13]
   + 2.392175912857056 Conv_1_aggregated_features[8,14]
   + 0.3933185040950775 Conv_1_aggregated_features[8,15] + Conv_1_t[8,0]
   = -2.192610263824463
 Conv_1_output_constraint[8,1]: 0.5202509164810181 Conv_0_Relu_ts[8,0]
   + 0.6686695218086243 Conv_0_Relu_ts[8,1]
   - 0.4720649719238281 Conv_0_Relu_ts[8,2]
   + 0.1026173532009125 Conv_0_Relu_ts[8,3]
   + 0.2593795359134674 Conv_0_Relu_ts[8,4]
   - 2.458898544311523 Conv_0_Relu_ts[8,5]
   - 0.5632196664810181 Conv_0_Relu_ts[8,6]
   - 1.791601657867432 Conv_0_Relu_ts[8,7]
   - 0.1268838942050934 Conv_0_Relu_ts[8,8]
   + 0.7120329737663269 Conv_0_Relu_ts[8,9]
   - 0.5212016701698303 Conv_0_Relu_ts[8,10]
   + 0.6136131286621094 Conv_0_Relu_ts[8,11]
   + 1.563590168952942 Conv_0_Relu_ts[8,12]
   - 0.3512922525405884 Conv_0_Relu_ts[8,13]
   + 0.5421575903892517 Conv_0_Relu_ts[8,14]
   + 0.7331249117851257 Conv_0_Relu_ts[8,15]
   + 0.0113700022920966 Conv_1_aggregated_features[8,0]
   + 2.169116258621216 Conv_1_aggregated_features[8,1]
   + 0.00623909616842866 Conv_1_aggregated_features[8,2]
   + 0.0937856733798981 Conv_1_aggregated_features[8,3]
   + 0.8328568339347839 Conv_1_aggregated_features[8,4]
   + 0.383074939250946 Conv_1_aggregated_features[8,5]
   - 0.1267539262771606 Conv_1_aggregated_features[8,6]
   + 0.6484018564224243 Conv_1_aggregated_features[8,7]
   + 0.1870926916599274 Conv_1_aggregated_features[8,8]
   - 0.4116691052913666 Conv_1_aggregated_features[8,9]
   + 0.1439376026391983 Conv_1_aggregated_features[8,10]
   + 1.05384635925293 Conv_1_aggregated_features[8,11]
   + 1.155439853668213 Conv_1_aggregated_features[8,12]
   - 0.2060132175683975 Conv_1_aggregated_features[8,13]
   + 0.6003169417381287 Conv_1_aggregated_features[8,14]
   + 0.1458597332239151 Conv_1_aggregated_features[8,15] + Conv_1_t[8,1]
   = -0.1077260375022888
 Conv_1_output_constraint[8,2]: 0.4269110858440399 Conv_0_Relu_ts[8,0]
   - 1.097789287567139 Conv_0_Relu_ts[8,1]
   - 0.3380239307880402 Conv_0_Relu_ts[8,2]
   + 0.23756343126297 Conv_0_Relu_ts[8,3]
   - 0.337738424539566 Conv_0_Relu_ts[8,4]
   - 1.407688736915588 Conv_0_Relu_ts[8,5]
   - 0.7854894995689392 Conv_0_Relu_ts[8,6]
   - 1.080899715423584 Conv_0_Relu_ts[8,7]
   + 0.0233876463025808 Conv_0_Relu_ts[8,8]
   + 1.849825739860535 Conv_0_Relu_ts[8,9]
   + 0.4580322206020355 Conv_0_Relu_ts[8,10]
   - 0.819361686706543 Conv_0_Relu_ts[8,11]
   - 0.0822571739554405 Conv_0_Relu_ts[8,12]
   - 0.1501271426677704 Conv_0_Relu_ts[8,13]
   + 0.1831801384687424 Conv_0_Relu_ts[8,14]
   + 0.00308510870672762 Conv_0_Relu_ts[8,15]
   - 0.2153502702713013 Conv_1_aggregated_features[8,0]
   + 3.817557334899902 Conv_1_aggregated_features[8,1]
   - 0.3402683734893799 Conv_1_aggregated_features[8,2]
   + 0.0650523602962494 Conv_1_aggregated_features[8,3]
   - 0.4435569643974304 Conv_1_aggregated_features[8,4]
   + 0.045248981565237 Conv_1_aggregated_features[8,5]
   + 0.844337522983551 Conv_1_aggregated_features[8,6]
   - 0.0691490024328232 Conv_1_aggregated_features[8,7]
   + 0.0830505788326263 Conv_1_aggregated_features[8,8]
   - 0.3022302687168121 Conv_1_aggregated_features[8,9]
   - 0.611566960811615 Conv_1_aggregated_features[8,10]
   + 3.582128524780273 Conv_1_aggregated_features[8,11]
   + 0.1314867585897446 Conv_1_aggregated_features[8,12]
   + 0.5333523154258728 Conv_1_aggregated_features[8,13]
   - 0.1983102858066559 Conv_1_aggregated_features[8,14]
   + 0.1628565639257431 Conv_1_aggregated_features[8,15] + Conv_1_t[8,2]
   = -0.4443503022193909
 Conv_1_output_constraint[8,3]: 0.066770151257515 Conv_0_Relu_ts[8,0]
   + 0.3136637210845947 Conv_0_Relu_ts[8,1]
   + 0.2078424692153931 Conv_0_Relu_ts[8,2]
   - 0.1673517823219299 Conv_0_Relu_ts[8,3]
   + 1.022224068641663 Conv_0_Relu_ts[8,4]
   - 0.1723377108573914 Conv_0_Relu_ts[8,5]
   + 0.3032898306846619 Conv_0_Relu_ts[8,6]
   + 0.0841449126601219 Conv_0_Relu_ts[8,7]
   + 0.0921086072921753 Conv_0_Relu_ts[8,8]
   - 0.022469462826848 Conv_0_Relu_ts[8,9]
   - 0.1657154262065887 Conv_0_Relu_ts[8,10]
   + 1.007031679153442 Conv_0_Relu_ts[8,11]
   + 0.0208180602639914 Conv_0_Relu_ts[8,12]
   + 0.3933880031108856 Conv_0_Relu_ts[8,13]
   + 0.3077191412448883 Conv_0_Relu_ts[8,14]
   + 0.0400979965925217 Conv_0_Relu_ts[8,15]
   + 0.0230858232825994 Conv_1_aggregated_features[8,0]
   + 0.3792195022106171 Conv_1_aggregated_features[8,1]
   + 0.1572305560112 Conv_1_aggregated_features[8,2]
   - 0.0165807008743286 Conv_1_aggregated_features[8,3]
   + 0.5649795532226562 Conv_1_aggregated_features[8,4]
   + 0.6253551840782166 Conv_1_aggregated_features[8,5]
   + 0.3968201577663422 Conv_1_aggregated_features[8,6]
   + 0.2728661298751831 Conv_1_aggregated_features[8,7]
   - 0.210608571767807 Conv_1_aggregated_features[8,8]
   + 0.094737209379673 Conv_1_aggregated_features[8,9]
   - 0.1354089975357056 Conv_1_aggregated_features[8,10]
   + 0.9330839514732361 Conv_1_aggregated_features[8,11]
   + 0.1677136421203613 Conv_1_aggregated_features[8,12]
   + 0.6573846936225891 Conv_1_aggregated_features[8,13]
   + 0.044663343578577 Conv_1_aggregated_features[8,14]
   - 0.0132347755134106 Conv_1_aggregated_features[8,15] + Conv_1_t[8,3]
   = -0.327608048915863
 Conv_1_output_constraint[8,4]: - 0.0720538049936295 Conv_0_Relu_ts[8,0]
   + 3.730369091033936 Conv_0_Relu_ts[8,1]
   - 0.8118388652801514 Conv_0_Relu_ts[8,2]
   - 0.174347460269928 Conv_0_Relu_ts[8,3]
   - 3.084089279174805 Conv_0_Relu_ts[8,4]
   - 5.146248340606689 Conv_0_Relu_ts[8,5]
   - 2.440467357635498 Conv_0_Relu_ts[8,6]
   - 3.765322923660278 Conv_0_Relu_ts[8,7]
   + 0.153628796339035 Conv_0_Relu_ts[8,8]
   + 0.1013353765010834 Conv_0_Relu_ts[8,9]
   - 0.5803871154785156 Conv_0_Relu_ts[8,10]
   - 2.657181739807129 Conv_0_Relu_ts[8,11]
   + 0.6840037703514099 Conv_0_Relu_ts[8,12]
   - 1.184461355209351 Conv_0_Relu_ts[8,13]
   - 0.3772841989994049 Conv_0_Relu_ts[8,14]
   + 0.5856359601020813 Conv_0_Relu_ts[8,15]
   + 0.2062403708696365 Conv_1_aggregated_features[8,0]
   + 1.470675468444824 Conv_1_aggregated_features[8,1]
   + 0.3523006737232208 Conv_1_aggregated_features[8,2]
   - 0.1351108849048615 Conv_1_aggregated_features[8,3]
   + 0.8932182788848877 Conv_1_aggregated_features[8,4]
   + 1.194340229034424 Conv_1_aggregated_features[8,5]
   + 0.0163087919354439 Conv_1_aggregated_features[8,6]
   - 0.5082771182060242 Conv_1_aggregated_features[8,7]
   - 0.0201514065265656 Conv_1_aggregated_features[8,8]
   + 0.9045064449310303 Conv_1_aggregated_features[8,9]
   + 0.7243757843971252 Conv_1_aggregated_features[8,10]
   + 0.5823414921760559 Conv_1_aggregated_features[8,11]
   + 0.0738087296485901 Conv_1_aggregated_features[8,12]
   - 0.3080458641052246 Conv_1_aggregated_features[8,13]
   + 0.9457569122314453 Conv_1_aggregated_features[8,14]
   + 0.4423830211162567 Conv_1_aggregated_features[8,15] + Conv_1_t[8,4]
   = -0.2138544023036957
 Conv_1_output_constraint[8,5]: 0.2116706073284149 Conv_0_Relu_ts[8,0]
   + 0.2319677472114563 Conv_0_Relu_ts[8,1]
   + 0.1967353820800781 Conv_0_Relu_ts[8,2]
   - 0.1974936425685883 Conv_0_Relu_ts[8,3]
   + 0.087493434548378 Conv_0_Relu_ts[8,4]
   + 0.0226023308932781 Conv_0_Relu_ts[8,5]
   - 0.1455883532762527 Conv_0_Relu_ts[8,6]
   - 0.1098284423351288 Conv_0_Relu_ts[8,7]
   + 0.0375414304435253 Conv_0_Relu_ts[8,8]
   + 0.2209681868553162 Conv_0_Relu_ts[8,9]
   + 0.0180207490921021 Conv_0_Relu_ts[8,10]
   - 0.0944192707538605 Conv_0_Relu_ts[8,11]
   + 0.1318628787994385 Conv_0_Relu_ts[8,12]
   + 0.1718421876430511 Conv_0_Relu_ts[8,13]
   - 0.0992768108844757 Conv_0_Relu_ts[8,14]
   + 0.2220577001571655 Conv_0_Relu_ts[8,15]
   + 0.0500434041023254 Conv_1_aggregated_features[8,0]
   + 0.2417693138122559 Conv_1_aggregated_features[8,1]
   + 0.1884889900684357 Conv_1_aggregated_features[8,2]
   - 0.0459566116333008 Conv_1_aggregated_features[8,3]
   - 0.1095028519630432 Conv_1_aggregated_features[8,4]
   - 0.1494371891021729 Conv_1_aggregated_features[8,5]
   - 0.0601528584957123 Conv_1_aggregated_features[8,6]
   - 0.2431226968765259 Conv_1_aggregated_features[8,7]
   + 0.0709281861782074 Conv_1_aggregated_features[8,8]
   + 0.1859873235225677 Conv_1_aggregated_features[8,9]
   - 0.0025022029876709 Conv_1_aggregated_features[8,10]
   + 0.1165908575057983 Conv_1_aggregated_features[8,11]
   - 0.0946725904941559 Conv_1_aggregated_features[8,12]
   + 0.1679102778434753 Conv_1_aggregated_features[8,13]
   + 0.0888539850711823 Conv_1_aggregated_features[8,14]
   + 0.0481749773025513 Conv_1_aggregated_features[8,15] + Conv_1_t[8,5]
   = -0.2017861604690552
 Conv_1_output_constraint[8,6]: - 0.5161762237548828 Conv_0_Relu_ts[8,0]
   - 0.9039249420166016 Conv_0_Relu_ts[8,1]
   - 0.5091935992240906 Conv_0_Relu_ts[8,2]
   + 0.177293062210083 Conv_0_Relu_ts[8,3]
   + 1.534894466400146 Conv_0_Relu_ts[8,4]
   + 0.3099820613861084 Conv_0_Relu_ts[8,5]
   + 1.335353136062622 Conv_0_Relu_ts[8,6]
   - 0.3965733051300049 Conv_0_Relu_ts[8,7]
   + 0.0226722583174706 Conv_0_Relu_ts[8,8]
   - 0.0153551828116179 Conv_0_Relu_ts[8,9]
   - 0.188335582613945 Conv_0_Relu_ts[8,10]
   + 0.1465377509593964 Conv_0_Relu_ts[8,11]
   - 0.4378408491611481 Conv_0_Relu_ts[8,12]
   + 0.5950998067855835 Conv_0_Relu_ts[8,13]
   - 0.2773410975933075 Conv_0_Relu_ts[8,14]
   - 0.3855878710746765 Conv_0_Relu_ts[8,15]
   + 0.0145458607003093 Conv_1_aggregated_features[8,0]
   - 1.349490284919739 Conv_1_aggregated_features[8,1]
   - 0.6738612055778503 Conv_1_aggregated_features[8,2]
   - 0.1977581083774567 Conv_1_aggregated_features[8,3]
   + 3.616869688034058 Conv_1_aggregated_features[8,4]
   - 0.5846084952354431 Conv_1_aggregated_features[8,5]
   + 0.3911246955394745 Conv_1_aggregated_features[8,6]
   - 0.2603789269924164 Conv_1_aggregated_features[8,7]
   + 0.0530429184436798 Conv_1_aggregated_features[8,8]
   + 0.7039647698402405 Conv_1_aggregated_features[8,9]
   - 0.4948081970214844 Conv_1_aggregated_features[8,10]
   - 1.33747124671936 Conv_1_aggregated_features[8,11]
   + 0.4521820247173309 Conv_1_aggregated_features[8,12]
   + 2.879487276077271 Conv_1_aggregated_features[8,13]
   + 0.6494181156158447 Conv_1_aggregated_features[8,14]
   - 0.0441654324531555 Conv_1_aggregated_features[8,15] + Conv_1_t[8,6]
   = -1.141431212425232
 Conv_1_output_constraint[8,7]: 0.1232976838946342 Conv_0_Relu_ts[8,0]
   + 0.7059634327888489 Conv_0_Relu_ts[8,1]
   + 0.0986589044332504 Conv_0_Relu_ts[8,2]
   - 0.1159188747406006 Conv_0_Relu_ts[8,3]
   + 0.1207201555371284 Conv_0_Relu_ts[8,4]
   + 0.4950260519981384 Conv_0_Relu_ts[8,5]
   - 0.1985135078430176 Conv_0_Relu_ts[8,6]
   + 0.3846564292907715 Conv_0_Relu_ts[8,7]
   - 0.2091868817806244 Conv_0_Relu_ts[8,8]
   + 0.0282646510750055 Conv_0_Relu_ts[8,9]
   + 0.00616153934970498 Conv_0_Relu_ts[8,10]
   - 0.2992357313632965 Conv_0_Relu_ts[8,11]
   + 0.6717308759689331 Conv_0_Relu_ts[8,12]
   + 0.3884595632553101 Conv_0_Relu_ts[8,13]
   + 0.1088093742728233 Conv_0_Relu_ts[8,14]
   + 0.1363743990659714 Conv_0_Relu_ts[8,15]
   + 0.2181874513626099 Conv_1_aggregated_features[8,0]
   + 0.1884859055280685 Conv_1_aggregated_features[8,1]
   + 0.1986039727926254 Conv_1_aggregated_features[8,2]
   - 0.0569350719451904 Conv_1_aggregated_features[8,3]
   + 0.4957895576953888 Conv_1_aggregated_features[8,4]
   - 0.2215360552072525 Conv_1_aggregated_features[8,5]
   - 0.3126825392246246 Conv_1_aggregated_features[8,6]
   + 0.2129529714584351 Conv_1_aggregated_features[8,7]
   + 0.1905750632286072 Conv_1_aggregated_features[8,8]
   + 0.3835417926311493 Conv_1_aggregated_features[8,9]
   + 0.0515264272689819 Conv_1_aggregated_features[8,10]
   + 0.2774861454963684 Conv_1_aggregated_features[8,11]
   + 0.1720549166202545 Conv_1_aggregated_features[8,12]
   + 0.2042496353387833 Conv_1_aggregated_features[8,13]
   + 0.1717791706323624 Conv_1_aggregated_features[8,14]
   - 0.0635386854410172 Conv_1_aggregated_features[8,15] + Conv_1_t[8,7]
   = -0.1961010098457336
 Conv_1_output_constraint[8,8]: - 0.0984712317585945 Conv_0_Relu_ts[8,0]
   - 0.8972246050834656 Conv_0_Relu_ts[8,1]
   - 0.1342054009437561 Conv_0_Relu_ts[8,2]
   + 0.0934775173664093 Conv_0_Relu_ts[8,3]
   + 3.929172039031982 Conv_0_Relu_ts[8,4]
   - 1.699812531471252 Conv_0_Relu_ts[8,5]
   + 0.4780299663543701 Conv_0_Relu_ts[8,6]
   + 0.1916014850139618 Conv_0_Relu_ts[8,7]
   - 0.0282735247164965 Conv_0_Relu_ts[8,8]
   - 0.10334312915802 Conv_0_Relu_ts[8,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[8,10]
   - 0.9000989198684692 Conv_0_Relu_ts[8,11]
   - 0.3381493389606476 Conv_0_Relu_ts[8,12]
   + 0.2327869236469269 Conv_0_Relu_ts[8,13]
   + 0.6162272691726685 Conv_0_Relu_ts[8,14]
   - 0.1045716181397438 Conv_0_Relu_ts[8,15]
   - 0.0895406752824783 Conv_1_aggregated_features[8,0]
   - 0.0733596011996269 Conv_1_aggregated_features[8,1]
   - 0.6064443588256836 Conv_1_aggregated_features[8,2]
   + 0.1685465276241302 Conv_1_aggregated_features[8,3]
   + 2.521227836608887 Conv_1_aggregated_features[8,4]
   + 0.8868938088417053 Conv_1_aggregated_features[8,5]
   - 0.3730944097042084 Conv_1_aggregated_features[8,6]
   + 0.4433300495147705 Conv_1_aggregated_features[8,7]
   + 0.2465403079986572 Conv_1_aggregated_features[8,8]
   + 0.1199371814727783 Conv_1_aggregated_features[8,9]
   - 0.427333652973175 Conv_1_aggregated_features[8,10]
   - 1.286244511604309 Conv_1_aggregated_features[8,11]
   + 0.4271916449069977 Conv_1_aggregated_features[8,12]
   + 1.475327014923096 Conv_1_aggregated_features[8,13]
   - 0.2351342439651489 Conv_1_aggregated_features[8,14]
   - 0.1273849010467529 Conv_1_aggregated_features[8,15] + Conv_1_t[8,8]
   = -1.303357601165771
 Conv_1_output_constraint[8,9]: 0.1534582376480103 Conv_0_Relu_ts[8,0]
   + 1.456407189369202 Conv_0_Relu_ts[8,1]
   - 0.0228729248046875 Conv_0_Relu_ts[8,2]
   + 0.1227743625640869 Conv_0_Relu_ts[8,3]
   + 0.7793847918510437 Conv_0_Relu_ts[8,4]
   - 0.2757052779197693 Conv_0_Relu_ts[8,5]
   + 0.6606881022453308 Conv_0_Relu_ts[8,6]
   - 1.107266664505005 Conv_0_Relu_ts[8,7]
   + 0.1723826229572296 Conv_0_Relu_ts[8,8]
   - 0.5540673732757568 Conv_0_Relu_ts[8,9]
   + 0.3133971095085144 Conv_0_Relu_ts[8,10]
   + 0.9318023324012756 Conv_0_Relu_ts[8,11]
   + 0.5855908989906311 Conv_0_Relu_ts[8,12]
   - 0.2116718590259552 Conv_0_Relu_ts[8,13]
   - 0.3693177700042725 Conv_0_Relu_ts[8,14]
   - 0.0642165616154671 Conv_0_Relu_ts[8,15]
   + 0.1819062232971191 Conv_1_aggregated_features[8,0]
   + 1.311912417411804 Conv_1_aggregated_features[8,1]
   + 0.3460695445537567 Conv_1_aggregated_features[8,2]
   + 0.1086616516113281 Conv_1_aggregated_features[8,3]
   - 0.6099227070808411 Conv_1_aggregated_features[8,4]
   + 0.3786990940570831 Conv_1_aggregated_features[8,5]
   + 0.9473716020584106 Conv_1_aggregated_features[8,6]
   + 0.2196423262357712 Conv_1_aggregated_features[8,7]
   + 0.0830722749233246 Conv_1_aggregated_features[8,8]
   - 0.4822347462177277 Conv_1_aggregated_features[8,9]
   + 0.5108910202980042 Conv_1_aggregated_features[8,10]
   + 3.214276790618896 Conv_1_aggregated_features[8,11]
   - 0.7224436402320862 Conv_1_aggregated_features[8,12]
   + 0.2678262591362 Conv_1_aggregated_features[8,13]
   - 0.8807565569877625 Conv_1_aggregated_features[8,14]
   - 0.3268090188503265 Conv_1_aggregated_features[8,15] + Conv_1_t[8,9]
   = -0.8283404111862183
 Conv_1_output_constraint[8,10]: 0.3653882741928101 Conv_0_Relu_ts[8,0]
   - 0.8208209872245789 Conv_0_Relu_ts[8,1]
   + 0.3088222146034241 Conv_0_Relu_ts[8,2]
   + 0.1173163652420044 Conv_0_Relu_ts[8,3]
   + 0.658578634262085 Conv_0_Relu_ts[8,4]
   + 0.584019660949707 Conv_0_Relu_ts[8,5]
   - 0.8654046058654785 Conv_0_Relu_ts[8,6]
   + 0.3697350919246674 Conv_0_Relu_ts[8,7]
   - 0.2274059355258942 Conv_0_Relu_ts[8,8]
   + 1.197938442230225 Conv_0_Relu_ts[8,9]
   + 0.2446341812610626 Conv_0_Relu_ts[8,10]
   - 0.5676743984222412 Conv_0_Relu_ts[8,11]
   - 0.3123646676540375 Conv_0_Relu_ts[8,12]
   + 3.181503534317017 Conv_0_Relu_ts[8,13]
   + 0.6926240921020508 Conv_0_Relu_ts[8,14]
   + 0.0886927545070648 Conv_0_Relu_ts[8,15]
   - 0.2005636394023895 Conv_1_aggregated_features[8,0]
   + 1.403087973594666 Conv_1_aggregated_features[8,1]
   + 0.0523448958992958 Conv_1_aggregated_features[8,2]
   + 0.2411543428897858 Conv_1_aggregated_features[8,3]
   + 0.5953171253204346 Conv_1_aggregated_features[8,4]
   - 0.8278964757919312 Conv_1_aggregated_features[8,5]
   + 0.2499217540025711 Conv_1_aggregated_features[8,6]
   - 0.7393644452095032 Conv_1_aggregated_features[8,7]
   + 0.0902875959873199 Conv_1_aggregated_features[8,8]
   + 0.1488711386919022 Conv_1_aggregated_features[8,9]
   - 0.0303693022578955 Conv_1_aggregated_features[8,10]
   + 2.024244070053101 Conv_1_aggregated_features[8,11]
   + 0.495431512594223 Conv_1_aggregated_features[8,12]
   + 0.3463623225688934 Conv_1_aggregated_features[8,13]
   - 0.0763808861374855 Conv_1_aggregated_features[8,14]
   + 0.1597562879323959 Conv_1_aggregated_features[8,15] + Conv_1_t[8,10]
   = -0.5063248872756958
 Conv_1_output_constraint[8,11]: - 0.5185666680335999 Conv_0_Relu_ts[8,0]
   + 3.656918525695801 Conv_0_Relu_ts[8,1]
   - 0.7082476615905762 Conv_0_Relu_ts[8,2]
   + 0.1922211349010468 Conv_0_Relu_ts[8,3]
   - 0.7149283289909363 Conv_0_Relu_ts[8,4]
   - 3.255658626556396 Conv_0_Relu_ts[8,5]
   + 2.464392185211182 Conv_0_Relu_ts[8,6]
   - 0.9477378129959106 Conv_0_Relu_ts[8,7]
   + 0.168237030506134 Conv_0_Relu_ts[8,8]
   - 1.177640557289124 Conv_0_Relu_ts[8,9]
   - 0.7062549591064453 Conv_0_Relu_ts[8,10]
   + 1.453806400299072 Conv_0_Relu_ts[8,11]
   + 0.2612711191177368 Conv_0_Relu_ts[8,12]
   + 0.2999382615089417 Conv_0_Relu_ts[8,13]
   - 1.255579590797424 Conv_0_Relu_ts[8,14]
   - 0.4265820384025574 Conv_0_Relu_ts[8,15]
   + 0.3379200994968414 Conv_1_aggregated_features[8,0]
   - 1.33229672908783 Conv_1_aggregated_features[8,1]
   - 0.1594242304563522 Conv_1_aggregated_features[8,2]
   + 0.2298935651779175 Conv_1_aggregated_features[8,3]
   - 1.311494588851929 Conv_1_aggregated_features[8,4]
   + 3.605161666870117 Conv_1_aggregated_features[8,5]
   - 0.3176752626895905 Conv_1_aggregated_features[8,6]
   + 0.3926282525062561 Conv_1_aggregated_features[8,7]
   + 0.1479305028915405 Conv_1_aggregated_features[8,8]
   - 0.0372937023639679 Conv_1_aggregated_features[8,9]
   + 0.3455427587032318 Conv_1_aggregated_features[8,10]
   - 2.002299070358276 Conv_1_aggregated_features[8,11]
   - 0.9904713034629822 Conv_1_aggregated_features[8,12]
   + 0.0751789212226868 Conv_1_aggregated_features[8,13]
   - 0.5166233777999878 Conv_1_aggregated_features[8,14]
   + 0.0510688796639442 Conv_1_aggregated_features[8,15] + Conv_1_t[8,11]
   = -1.480785727500916
 Conv_1_output_constraint[8,12]: 0.3080185949802399 Conv_0_Relu_ts[8,0]
   + 0.0994751453399658 Conv_0_Relu_ts[8,1]
   + 0.4488444924354553 Conv_0_Relu_ts[8,2]
   + 0.0305773317813873 Conv_0_Relu_ts[8,3]
   - 1.673393130302429 Conv_0_Relu_ts[8,4]
   + 0.2824500501155853 Conv_0_Relu_ts[8,5]
   - 1.192709684371948 Conv_0_Relu_ts[8,6]
   + 1.454243063926697 Conv_0_Relu_ts[8,7]
   + 0.1514801830053329 Conv_0_Relu_ts[8,8]
   + 1.026351928710938 Conv_0_Relu_ts[8,9]
   + 0.1183762699365616 Conv_0_Relu_ts[8,10]
   + 0.352224588394165 Conv_0_Relu_ts[8,11]
   + 1.090276837348938 Conv_0_Relu_ts[8,12]
   - 2.053535461425781 Conv_0_Relu_ts[8,13]
   + 2.538641452789307 Conv_0_Relu_ts[8,14]
   + 0.6387560963630676 Conv_0_Relu_ts[8,15]
   - 0.1990496516227722 Conv_1_aggregated_features[8,0]
   + 1.247365713119507 Conv_1_aggregated_features[8,1]
   + 0.2213435918092728 Conv_1_aggregated_features[8,2]
   + 0.2398849725723267 Conv_1_aggregated_features[8,3]
   - 2.001479864120483 Conv_1_aggregated_features[8,4]
   - 1.084800243377686 Conv_1_aggregated_features[8,5]
   + 0.4287458956241608 Conv_1_aggregated_features[8,6]
   - 0.8179780840873718 Conv_1_aggregated_features[8,7]
   - 0.0883614718914032 Conv_1_aggregated_features[8,8]
   - 0.4398066997528076 Conv_1_aggregated_features[8,9]
   + 0.2760448455810547 Conv_1_aggregated_features[8,10]
   + 1.944519639015198 Conv_1_aggregated_features[8,11]
   + 1.041437149047852 Conv_1_aggregated_features[8,12]
   - 1.682050704956055 Conv_1_aggregated_features[8,13]
   - 0.1825132220983505 Conv_1_aggregated_features[8,14]
   + 0.2881160080432892 Conv_1_aggregated_features[8,15] + Conv_1_t[8,12]
   = 2.457676649093628
 Conv_1_output_constraint[8,13]: 0.2853958606719971 Conv_0_Relu_ts[8,0]
   + 0.0445350483059883 Conv_0_Relu_ts[8,1]
   - 0.0597315393388271 Conv_0_Relu_ts[8,2]
   + 0.2403226792812347 Conv_0_Relu_ts[8,3]
   + 0.5103238224983215 Conv_0_Relu_ts[8,4]
   - 0.1369927227497101 Conv_0_Relu_ts[8,5]
   + 0.5091087818145752 Conv_0_Relu_ts[8,6]
   - 0.0576926171779633 Conv_0_Relu_ts[8,7]
   - 0.00701320171356201 Conv_0_Relu_ts[8,8]
   + 0.5782948136329651 Conv_0_Relu_ts[8,9]
   - 0.1188536584377289 Conv_0_Relu_ts[8,10]
   - 0.078026682138443 Conv_0_Relu_ts[8,11]
   + 0.1965750753879547 Conv_0_Relu_ts[8,12]
   + 0.4982857704162598 Conv_0_Relu_ts[8,13]
   - 0.0995257869362831 Conv_0_Relu_ts[8,14]
   + 0.0712818428874016 Conv_0_Relu_ts[8,15]
   + 0.2628078460693359 Conv_1_aggregated_features[8,0]
   + 0.6577805280685425 Conv_1_aggregated_features[8,1]
   - 0.1415485441684723 Conv_1_aggregated_features[8,2]
   + 0.1597791016101837 Conv_1_aggregated_features[8,3]
   + 0.3969745337963104 Conv_1_aggregated_features[8,4]
   - 0.3941676616668701 Conv_1_aggregated_features[8,5]
   + 0.3484204709529877 Conv_1_aggregated_features[8,6]
   + 0.2601790428161621 Conv_1_aggregated_features[8,7]
   - 0.0652478933334351 Conv_1_aggregated_features[8,8]
   + 0.3492280542850494 Conv_1_aggregated_features[8,9]
   + 0.1917867064476013 Conv_1_aggregated_features[8,10]
   + 0.2439181953668594 Conv_1_aggregated_features[8,11]
   + 0.1009163707494736 Conv_1_aggregated_features[8,12]
   + 0.2350947707891464 Conv_1_aggregated_features[8,13]
   + 0.2627962231636047 Conv_1_aggregated_features[8,14]
   - 0.0959920063614845 Conv_1_aggregated_features[8,15] + Conv_1_t[8,13]
   = -0.4268372058868408
 Conv_1_output_constraint[8,14]: - 0.2448655962944031 Conv_0_Relu_ts[8,0]
   + 0.2688764035701752 Conv_0_Relu_ts[8,1]
   + 0.0118266940116882 Conv_0_Relu_ts[8,2]
   - 0.0670525431632996 Conv_0_Relu_ts[8,3]
   + 0.1511579602956772 Conv_0_Relu_ts[8,4]
   - 0.4796198606491089 Conv_0_Relu_ts[8,5]
   - 0.1699391901493073 Conv_0_Relu_ts[8,6]
   + 0.2647148370742798 Conv_0_Relu_ts[8,7]
   + 0.137117475271225 Conv_0_Relu_ts[8,8]
   + 0.1737731248140335 Conv_0_Relu_ts[8,9]
   - 0.1610838174819946 Conv_0_Relu_ts[8,10]
   - 0.6311220526695251 Conv_0_Relu_ts[8,11]
   + 0.6263757348060608 Conv_0_Relu_ts[8,12]
   - 0.4823384284973145 Conv_0_Relu_ts[8,13]
   + 0.2192765176296234 Conv_0_Relu_ts[8,14]
   + 0.1739131808280945 Conv_0_Relu_ts[8,15]
   + 0.0443203188478947 Conv_1_aggregated_features[8,0]
   + 0.7668815851211548 Conv_1_aggregated_features[8,1]
   + 0.248365581035614 Conv_1_aggregated_features[8,2]
   - 0.2433404624462128 Conv_1_aggregated_features[8,3]
   + 0.2770798802375793 Conv_1_aggregated_features[8,4]
   - 1.224791765213013 Conv_1_aggregated_features[8,5]
   - 0.1892879456281662 Conv_1_aggregated_features[8,6]
   + 0.1141589805483818 Conv_1_aggregated_features[8,7]
   + 0.2022489905357361 Conv_1_aggregated_features[8,8]
   - 0.0391396470367908 Conv_1_aggregated_features[8,9]
   - 0.0935317650437355 Conv_1_aggregated_features[8,10]
   - 0.5407891273498535 Conv_1_aggregated_features[8,11]
   + 0.3130630552768707 Conv_1_aggregated_features[8,12]
   + 0.3410882651805878 Conv_1_aggregated_features[8,13]
   - 1.855417490005493 Conv_1_aggregated_features[8,14]
   + 0.247613787651062 Conv_1_aggregated_features[8,15] + Conv_1_t[8,14]
   = -0.2306040376424789
 Conv_1_output_constraint[8,15]: 0.043729841709137 Conv_0_Relu_ts[8,0]
   + 0.045797374099493 Conv_0_Relu_ts[8,1]
   - 0.1762047111988068 Conv_0_Relu_ts[8,2]
   - 0.2225021123886108 Conv_0_Relu_ts[8,3]
   + 0.1344186365604401 Conv_0_Relu_ts[8,4]
   + 0.2975046932697296 Conv_0_Relu_ts[8,5]
   + 0.1135610193014145 Conv_0_Relu_ts[8,6]
   + 0.165085032582283 Conv_0_Relu_ts[8,7]
   + 0.0632660016417503 Conv_0_Relu_ts[8,8]
   + 0.2534148395061493 Conv_0_Relu_ts[8,9]
   + 0.1100017726421356 Conv_0_Relu_ts[8,10]
   + 0.045901358127594 Conv_0_Relu_ts[8,11]
   + 0.2102706432342529 Conv_0_Relu_ts[8,12]
   + 0.1767551600933075 Conv_0_Relu_ts[8,13]
   - 0.0864512398838997 Conv_0_Relu_ts[8,14]
   + 0.1616824418306351 Conv_0_Relu_ts[8,15]
   + 0.0113421082496643 Conv_1_aggregated_features[8,0]
   + 0.2387615889310837 Conv_1_aggregated_features[8,1]
   + 0.1763307154178619 Conv_1_aggregated_features[8,2]
   + 0.1442219316959381 Conv_1_aggregated_features[8,3]
   + 0.3573774099349976 Conv_1_aggregated_features[8,4]
   - 0.0406427308917046 Conv_1_aggregated_features[8,5]
   + 0.1155626475811005 Conv_1_aggregated_features[8,6]
   + 0.0186243671923876 Conv_1_aggregated_features[8,7]
   - 0.1389756500720978 Conv_1_aggregated_features[8,8]
   + 0.274962306022644 Conv_1_aggregated_features[8,9]
   + 0.0309249386191368 Conv_1_aggregated_features[8,10]
   - 0.1606657803058624 Conv_1_aggregated_features[8,11]
   + 0.2456246316432953 Conv_1_aggregated_features[8,12]
   + 0.4163841009140015 Conv_1_aggregated_features[8,13]
   + 0.29454505443573 Conv_1_aggregated_features[8,14]
   + 0.1688967496156693 Conv_1_aggregated_features[8,15] + Conv_1_t[8,15]
   = -0.112743653357029
 Conv_1_output_constraint[9,0]: - 0.3061666786670685 Conv_0_Relu_ts[9,0]
   + 0.8854666948318481 Conv_0_Relu_ts[9,1]
   - 1.108363628387451 Conv_0_Relu_ts[9,2]
   - 0.0485447347164154 Conv_0_Relu_ts[9,3]
   - 0.3054571747779846 Conv_0_Relu_ts[9,4]
   - 1.666001558303833 Conv_0_Relu_ts[9,5]
   + 1.575538277626038 Conv_0_Relu_ts[9,6]
   - 0.8786032199859619 Conv_0_Relu_ts[9,7]
   + 0.1696695387363434 Conv_0_Relu_ts[9,8]
   - 0.2058112174272537 Conv_0_Relu_ts[9,9]
   - 0.8619717359542847 Conv_0_Relu_ts[9,10]
   + 1.268710613250732 Conv_0_Relu_ts[9,11]
   + 0.1005195453763008 Conv_0_Relu_ts[9,12]
   - 0.00627355137839913 Conv_0_Relu_ts[9,13]
   - 0.6727985143661499 Conv_0_Relu_ts[9,14]
   - 0.6082888841629028 Conv_0_Relu_ts[9,15]
   + 0.0746944323182106 Conv_1_aggregated_features[9,0]
   - 0.5317568182945251 Conv_1_aggregated_features[9,1]
   + 0.4405037760734558 Conv_1_aggregated_features[9,2]
   - 0.1488203704357147 Conv_1_aggregated_features[9,3]
   + 2.01956033706665 Conv_1_aggregated_features[9,4]
   + 0.8369019031524658 Conv_1_aggregated_features[9,5]
   + 0.1561367213726044 Conv_1_aggregated_features[9,6]
   + 0.1698544919490814 Conv_1_aggregated_features[9,7]
   - 0.0660476386547089 Conv_1_aggregated_features[9,8]
   + 1.570046067237854 Conv_1_aggregated_features[9,9]
   + 0.5540977120399475 Conv_1_aggregated_features[9,10]
   - 1.275410532951355 Conv_1_aggregated_features[9,11]
   - 0.092900924384594 Conv_1_aggregated_features[9,12]
   + 3.287018537521362 Conv_1_aggregated_features[9,13]
   + 2.392175912857056 Conv_1_aggregated_features[9,14]
   + 0.3933185040950775 Conv_1_aggregated_features[9,15] + Conv_1_t[9,0]
   = -2.192610263824463
 Conv_1_output_constraint[9,1]: 0.5202509164810181 Conv_0_Relu_ts[9,0]
   + 0.6686695218086243 Conv_0_Relu_ts[9,1]
   - 0.4720649719238281 Conv_0_Relu_ts[9,2]
   + 0.1026173532009125 Conv_0_Relu_ts[9,3]
   + 0.2593795359134674 Conv_0_Relu_ts[9,4]
   - 2.458898544311523 Conv_0_Relu_ts[9,5]
   - 0.5632196664810181 Conv_0_Relu_ts[9,6]
   - 1.791601657867432 Conv_0_Relu_ts[9,7]
   - 0.1268838942050934 Conv_0_Relu_ts[9,8]
   + 0.7120329737663269 Conv_0_Relu_ts[9,9]
   - 0.5212016701698303 Conv_0_Relu_ts[9,10]
   + 0.6136131286621094 Conv_0_Relu_ts[9,11]
   + 1.563590168952942 Conv_0_Relu_ts[9,12]
   - 0.3512922525405884 Conv_0_Relu_ts[9,13]
   + 0.5421575903892517 Conv_0_Relu_ts[9,14]
   + 0.7331249117851257 Conv_0_Relu_ts[9,15]
   + 0.0113700022920966 Conv_1_aggregated_features[9,0]
   + 2.169116258621216 Conv_1_aggregated_features[9,1]
   + 0.00623909616842866 Conv_1_aggregated_features[9,2]
   + 0.0937856733798981 Conv_1_aggregated_features[9,3]
   + 0.8328568339347839 Conv_1_aggregated_features[9,4]
   + 0.383074939250946 Conv_1_aggregated_features[9,5]
   - 0.1267539262771606 Conv_1_aggregated_features[9,6]
   + 0.6484018564224243 Conv_1_aggregated_features[9,7]
   + 0.1870926916599274 Conv_1_aggregated_features[9,8]
   - 0.4116691052913666 Conv_1_aggregated_features[9,9]
   + 0.1439376026391983 Conv_1_aggregated_features[9,10]
   + 1.05384635925293 Conv_1_aggregated_features[9,11]
   + 1.155439853668213 Conv_1_aggregated_features[9,12]
   - 0.2060132175683975 Conv_1_aggregated_features[9,13]
   + 0.6003169417381287 Conv_1_aggregated_features[9,14]
   + 0.1458597332239151 Conv_1_aggregated_features[9,15] + Conv_1_t[9,1]
   = -0.1077260375022888
 Conv_1_output_constraint[9,2]: 0.4269110858440399 Conv_0_Relu_ts[9,0]
   - 1.097789287567139 Conv_0_Relu_ts[9,1]
   - 0.3380239307880402 Conv_0_Relu_ts[9,2]
   + 0.23756343126297 Conv_0_Relu_ts[9,3]
   - 0.337738424539566 Conv_0_Relu_ts[9,4]
   - 1.407688736915588 Conv_0_Relu_ts[9,5]
   - 0.7854894995689392 Conv_0_Relu_ts[9,6]
   - 1.080899715423584 Conv_0_Relu_ts[9,7]
   + 0.0233876463025808 Conv_0_Relu_ts[9,8]
   + 1.849825739860535 Conv_0_Relu_ts[9,9]
   + 0.4580322206020355 Conv_0_Relu_ts[9,10]
   - 0.819361686706543 Conv_0_Relu_ts[9,11]
   - 0.0822571739554405 Conv_0_Relu_ts[9,12]
   - 0.1501271426677704 Conv_0_Relu_ts[9,13]
   + 0.1831801384687424 Conv_0_Relu_ts[9,14]
   + 0.00308510870672762 Conv_0_Relu_ts[9,15]
   - 0.2153502702713013 Conv_1_aggregated_features[9,0]
   + 3.817557334899902 Conv_1_aggregated_features[9,1]
   - 0.3402683734893799 Conv_1_aggregated_features[9,2]
   + 0.0650523602962494 Conv_1_aggregated_features[9,3]
   - 0.4435569643974304 Conv_1_aggregated_features[9,4]
   + 0.045248981565237 Conv_1_aggregated_features[9,5]
   + 0.844337522983551 Conv_1_aggregated_features[9,6]
   - 0.0691490024328232 Conv_1_aggregated_features[9,7]
   + 0.0830505788326263 Conv_1_aggregated_features[9,8]
   - 0.3022302687168121 Conv_1_aggregated_features[9,9]
   - 0.611566960811615 Conv_1_aggregated_features[9,10]
   + 3.582128524780273 Conv_1_aggregated_features[9,11]
   + 0.1314867585897446 Conv_1_aggregated_features[9,12]
   + 0.5333523154258728 Conv_1_aggregated_features[9,13]
   - 0.1983102858066559 Conv_1_aggregated_features[9,14]
   + 0.1628565639257431 Conv_1_aggregated_features[9,15] + Conv_1_t[9,2]
   = -0.4443503022193909
 Conv_1_output_constraint[9,3]: 0.066770151257515 Conv_0_Relu_ts[9,0]
   + 0.3136637210845947 Conv_0_Relu_ts[9,1]
   + 0.2078424692153931 Conv_0_Relu_ts[9,2]
   - 0.1673517823219299 Conv_0_Relu_ts[9,3]
   + 1.022224068641663 Conv_0_Relu_ts[9,4]
   - 0.1723377108573914 Conv_0_Relu_ts[9,5]
   + 0.3032898306846619 Conv_0_Relu_ts[9,6]
   + 0.0841449126601219 Conv_0_Relu_ts[9,7]
   + 0.0921086072921753 Conv_0_Relu_ts[9,8]
   - 0.022469462826848 Conv_0_Relu_ts[9,9]
   - 0.1657154262065887 Conv_0_Relu_ts[9,10]
   + 1.007031679153442 Conv_0_Relu_ts[9,11]
   + 0.0208180602639914 Conv_0_Relu_ts[9,12]
   + 0.3933880031108856 Conv_0_Relu_ts[9,13]
   + 0.3077191412448883 Conv_0_Relu_ts[9,14]
   + 0.0400979965925217 Conv_0_Relu_ts[9,15]
   + 0.0230858232825994 Conv_1_aggregated_features[9,0]
   + 0.3792195022106171 Conv_1_aggregated_features[9,1]
   + 0.1572305560112 Conv_1_aggregated_features[9,2]
   - 0.0165807008743286 Conv_1_aggregated_features[9,3]
   + 0.5649795532226562 Conv_1_aggregated_features[9,4]
   + 0.6253551840782166 Conv_1_aggregated_features[9,5]
   + 0.3968201577663422 Conv_1_aggregated_features[9,6]
   + 0.2728661298751831 Conv_1_aggregated_features[9,7]
   - 0.210608571767807 Conv_1_aggregated_features[9,8]
   + 0.094737209379673 Conv_1_aggregated_features[9,9]
   - 0.1354089975357056 Conv_1_aggregated_features[9,10]
   + 0.9330839514732361 Conv_1_aggregated_features[9,11]
   + 0.1677136421203613 Conv_1_aggregated_features[9,12]
   + 0.6573846936225891 Conv_1_aggregated_features[9,13]
   + 0.044663343578577 Conv_1_aggregated_features[9,14]
   - 0.0132347755134106 Conv_1_aggregated_features[9,15] + Conv_1_t[9,3]
   = -0.327608048915863
 Conv_1_output_constraint[9,4]: - 0.0720538049936295 Conv_0_Relu_ts[9,0]
   + 3.730369091033936 Conv_0_Relu_ts[9,1]
   - 0.8118388652801514 Conv_0_Relu_ts[9,2]
   - 0.174347460269928 Conv_0_Relu_ts[9,3]
   - 3.084089279174805 Conv_0_Relu_ts[9,4]
   - 5.146248340606689 Conv_0_Relu_ts[9,5]
   - 2.440467357635498 Conv_0_Relu_ts[9,6]
   - 3.765322923660278 Conv_0_Relu_ts[9,7]
   + 0.153628796339035 Conv_0_Relu_ts[9,8]
   + 0.1013353765010834 Conv_0_Relu_ts[9,9]
   - 0.5803871154785156 Conv_0_Relu_ts[9,10]
   - 2.657181739807129 Conv_0_Relu_ts[9,11]
   + 0.6840037703514099 Conv_0_Relu_ts[9,12]
   - 1.184461355209351 Conv_0_Relu_ts[9,13]
   - 0.3772841989994049 Conv_0_Relu_ts[9,14]
   + 0.5856359601020813 Conv_0_Relu_ts[9,15]
   + 0.2062403708696365 Conv_1_aggregated_features[9,0]
   + 1.470675468444824 Conv_1_aggregated_features[9,1]
   + 0.3523006737232208 Conv_1_aggregated_features[9,2]
   - 0.1351108849048615 Conv_1_aggregated_features[9,3]
   + 0.8932182788848877 Conv_1_aggregated_features[9,4]
   + 1.194340229034424 Conv_1_aggregated_features[9,5]
   + 0.0163087919354439 Conv_1_aggregated_features[9,6]
   - 0.5082771182060242 Conv_1_aggregated_features[9,7]
   - 0.0201514065265656 Conv_1_aggregated_features[9,8]
   + 0.9045064449310303 Conv_1_aggregated_features[9,9]
   + 0.7243757843971252 Conv_1_aggregated_features[9,10]
   + 0.5823414921760559 Conv_1_aggregated_features[9,11]
   + 0.0738087296485901 Conv_1_aggregated_features[9,12]
   - 0.3080458641052246 Conv_1_aggregated_features[9,13]
   + 0.9457569122314453 Conv_1_aggregated_features[9,14]
   + 0.4423830211162567 Conv_1_aggregated_features[9,15] + Conv_1_t[9,4]
   = -0.2138544023036957
 Conv_1_output_constraint[9,5]: 0.2116706073284149 Conv_0_Relu_ts[9,0]
   + 0.2319677472114563 Conv_0_Relu_ts[9,1]
   + 0.1967353820800781 Conv_0_Relu_ts[9,2]
   - 0.1974936425685883 Conv_0_Relu_ts[9,3]
   + 0.087493434548378 Conv_0_Relu_ts[9,4]
   + 0.0226023308932781 Conv_0_Relu_ts[9,5]
   - 0.1455883532762527 Conv_0_Relu_ts[9,6]
   - 0.1098284423351288 Conv_0_Relu_ts[9,7]
   + 0.0375414304435253 Conv_0_Relu_ts[9,8]
   + 0.2209681868553162 Conv_0_Relu_ts[9,9]
   + 0.0180207490921021 Conv_0_Relu_ts[9,10]
   - 0.0944192707538605 Conv_0_Relu_ts[9,11]
   + 0.1318628787994385 Conv_0_Relu_ts[9,12]
   + 0.1718421876430511 Conv_0_Relu_ts[9,13]
   - 0.0992768108844757 Conv_0_Relu_ts[9,14]
   + 0.2220577001571655 Conv_0_Relu_ts[9,15]
   + 0.0500434041023254 Conv_1_aggregated_features[9,0]
   + 0.2417693138122559 Conv_1_aggregated_features[9,1]
   + 0.1884889900684357 Conv_1_aggregated_features[9,2]
   - 0.0459566116333008 Conv_1_aggregated_features[9,3]
   - 0.1095028519630432 Conv_1_aggregated_features[9,4]
   - 0.1494371891021729 Conv_1_aggregated_features[9,5]
   - 0.0601528584957123 Conv_1_aggregated_features[9,6]
   - 0.2431226968765259 Conv_1_aggregated_features[9,7]
   + 0.0709281861782074 Conv_1_aggregated_features[9,8]
   + 0.1859873235225677 Conv_1_aggregated_features[9,9]
   - 0.0025022029876709 Conv_1_aggregated_features[9,10]
   + 0.1165908575057983 Conv_1_aggregated_features[9,11]
   - 0.0946725904941559 Conv_1_aggregated_features[9,12]
   + 0.1679102778434753 Conv_1_aggregated_features[9,13]
   + 0.0888539850711823 Conv_1_aggregated_features[9,14]
   + 0.0481749773025513 Conv_1_aggregated_features[9,15] + Conv_1_t[9,5]
   = -0.2017861604690552
 Conv_1_output_constraint[9,6]: - 0.5161762237548828 Conv_0_Relu_ts[9,0]
   - 0.9039249420166016 Conv_0_Relu_ts[9,1]
   - 0.5091935992240906 Conv_0_Relu_ts[9,2]
   + 0.177293062210083 Conv_0_Relu_ts[9,3]
   + 1.534894466400146 Conv_0_Relu_ts[9,4]
   + 0.3099820613861084 Conv_0_Relu_ts[9,5]
   + 1.335353136062622 Conv_0_Relu_ts[9,6]
   - 0.3965733051300049 Conv_0_Relu_ts[9,7]
   + 0.0226722583174706 Conv_0_Relu_ts[9,8]
   - 0.0153551828116179 Conv_0_Relu_ts[9,9]
   - 0.188335582613945 Conv_0_Relu_ts[9,10]
   + 0.1465377509593964 Conv_0_Relu_ts[9,11]
   - 0.4378408491611481 Conv_0_Relu_ts[9,12]
   + 0.5950998067855835 Conv_0_Relu_ts[9,13]
   - 0.2773410975933075 Conv_0_Relu_ts[9,14]
   - 0.3855878710746765 Conv_0_Relu_ts[9,15]
   + 0.0145458607003093 Conv_1_aggregated_features[9,0]
   - 1.349490284919739 Conv_1_aggregated_features[9,1]
   - 0.6738612055778503 Conv_1_aggregated_features[9,2]
   - 0.1977581083774567 Conv_1_aggregated_features[9,3]
   + 3.616869688034058 Conv_1_aggregated_features[9,4]
   - 0.5846084952354431 Conv_1_aggregated_features[9,5]
   + 0.3911246955394745 Conv_1_aggregated_features[9,6]
   - 0.2603789269924164 Conv_1_aggregated_features[9,7]
   + 0.0530429184436798 Conv_1_aggregated_features[9,8]
   + 0.7039647698402405 Conv_1_aggregated_features[9,9]
   - 0.4948081970214844 Conv_1_aggregated_features[9,10]
   - 1.33747124671936 Conv_1_aggregated_features[9,11]
   + 0.4521820247173309 Conv_1_aggregated_features[9,12]
   + 2.879487276077271 Conv_1_aggregated_features[9,13]
   + 0.6494181156158447 Conv_1_aggregated_features[9,14]
   - 0.0441654324531555 Conv_1_aggregated_features[9,15] + Conv_1_t[9,6]
   = -1.141431212425232
 Conv_1_output_constraint[9,7]: 0.1232976838946342 Conv_0_Relu_ts[9,0]
   + 0.7059634327888489 Conv_0_Relu_ts[9,1]
   + 0.0986589044332504 Conv_0_Relu_ts[9,2]
   - 0.1159188747406006 Conv_0_Relu_ts[9,3]
   + 0.1207201555371284 Conv_0_Relu_ts[9,4]
   + 0.4950260519981384 Conv_0_Relu_ts[9,5]
   - 0.1985135078430176 Conv_0_Relu_ts[9,6]
   + 0.3846564292907715 Conv_0_Relu_ts[9,7]
   - 0.2091868817806244 Conv_0_Relu_ts[9,8]
   + 0.0282646510750055 Conv_0_Relu_ts[9,9]
   + 0.00616153934970498 Conv_0_Relu_ts[9,10]
   - 0.2992357313632965 Conv_0_Relu_ts[9,11]
   + 0.6717308759689331 Conv_0_Relu_ts[9,12]
   + 0.3884595632553101 Conv_0_Relu_ts[9,13]
   + 0.1088093742728233 Conv_0_Relu_ts[9,14]
   + 0.1363743990659714 Conv_0_Relu_ts[9,15]
   + 0.2181874513626099 Conv_1_aggregated_features[9,0]
   + 0.1884859055280685 Conv_1_aggregated_features[9,1]
   + 0.1986039727926254 Conv_1_aggregated_features[9,2]
   - 0.0569350719451904 Conv_1_aggregated_features[9,3]
   + 0.4957895576953888 Conv_1_aggregated_features[9,4]
   - 0.2215360552072525 Conv_1_aggregated_features[9,5]
   - 0.3126825392246246 Conv_1_aggregated_features[9,6]
   + 0.2129529714584351 Conv_1_aggregated_features[9,7]
   + 0.1905750632286072 Conv_1_aggregated_features[9,8]
   + 0.3835417926311493 Conv_1_aggregated_features[9,9]
   + 0.0515264272689819 Conv_1_aggregated_features[9,10]
   + 0.2774861454963684 Conv_1_aggregated_features[9,11]
   + 0.1720549166202545 Conv_1_aggregated_features[9,12]
   + 0.2042496353387833 Conv_1_aggregated_features[9,13]
   + 0.1717791706323624 Conv_1_aggregated_features[9,14]
   - 0.0635386854410172 Conv_1_aggregated_features[9,15] + Conv_1_t[9,7]
   = -0.1961010098457336
 Conv_1_output_constraint[9,8]: - 0.0984712317585945 Conv_0_Relu_ts[9,0]
   - 0.8972246050834656 Conv_0_Relu_ts[9,1]
   - 0.1342054009437561 Conv_0_Relu_ts[9,2]
   + 0.0934775173664093 Conv_0_Relu_ts[9,3]
   + 3.929172039031982 Conv_0_Relu_ts[9,4]
   - 1.699812531471252 Conv_0_Relu_ts[9,5]
   + 0.4780299663543701 Conv_0_Relu_ts[9,6]
   + 0.1916014850139618 Conv_0_Relu_ts[9,7]
   - 0.0282735247164965 Conv_0_Relu_ts[9,8]
   - 0.10334312915802 Conv_0_Relu_ts[9,9]
   + 5.6185130961239338e-04 Conv_0_Relu_ts[9,10]
   - 0.9000989198684692 Conv_0_Relu_ts[9,11]
   - 0.3381493389606476 Conv_0_Relu_ts[9,12]
   + 0.2327869236469269 Conv_0_Relu_ts[9,13]
   + 0.6162272691726685 Conv_0_Relu_ts[9,14]
   - 0.1045716181397438 Conv_0_Relu_ts[9,15]
   - 0.0895406752824783 Conv_1_aggregated_features[9,0]
   - 0.0733596011996269 Conv_1_aggregated_features[9,1]
   - 0.6064443588256836 Conv_1_aggregated_features[9,2]
   + 0.1685465276241302 Conv_1_aggregated_features[9,3]
   + 2.521227836608887 Conv_1_aggregated_features[9,4]
   + 0.8868938088417053 Conv_1_aggregated_features[9,5]
   - 0.3730944097042084 Conv_1_aggregated_features[9,6]
   + 0.4433300495147705 Conv_1_aggregated_features[9,7]
   + 0.2465403079986572 Conv_1_aggregated_features[9,8]
   + 0.1199371814727783 Conv_1_aggregated_features[9,9]
   - 0.427333652973175 Conv_1_aggregated_features[9,10]
   - 1.286244511604309 Conv_1_aggregated_features[9,11]
   + 0.4271916449069977 Conv_1_aggregated_features[9,12]
   + 1.475327014923096 Conv_1_aggregated_features[9,13]
   - 0.2351342439651489 Conv_1_aggregated_features[9,14]
   - 0.1273849010467529 Conv_1_aggregated_features[9,15] + Conv_1_t[9,8]
   = -1.303357601165771
 Conv_1_output_constraint[9,9]: 0.1534582376480103 Conv_0_Relu_ts[9,0]
   + 1.456407189369202 Conv_0_Relu_ts[9,1]
   - 0.0228729248046875 Conv_0_Relu_ts[9,2]
   + 0.1227743625640869 Conv_0_Relu_ts[9,3]
   + 0.7793847918510437 Conv_0_Relu_ts[9,4]
   - 0.2757052779197693 Conv_0_Relu_ts[9,5]
   + 0.6606881022453308 Conv_0_Relu_ts[9,6]
   - 1.107266664505005 Conv_0_Relu_ts[9,7]
   + 0.1723826229572296 Conv_0_Relu_ts[9,8]
   - 0.5540673732757568 Conv_0_Relu_ts[9,9]
   + 0.3133971095085144 Conv_0_Relu_ts[9,10]
   + 0.9318023324012756 Conv_0_Relu_ts[9,11]
   + 0.5855908989906311 Conv_0_Relu_ts[9,12]
   - 0.2116718590259552 Conv_0_Relu_ts[9,13]
   - 0.3693177700042725 Conv_0_Relu_ts[9,14]
   - 0.0642165616154671 Conv_0_Relu_ts[9,15]
   + 0.1819062232971191 Conv_1_aggregated_features[9,0]
   + 1.311912417411804 Conv_1_aggregated_features[9,1]
   + 0.3460695445537567 Conv_1_aggregated_features[9,2]
   + 0.1086616516113281 Conv_1_aggregated_features[9,3]
   - 0.6099227070808411 Conv_1_aggregated_features[9,4]
   + 0.3786990940570831 Conv_1_aggregated_features[9,5]
   + 0.9473716020584106 Conv_1_aggregated_features[9,6]
   + 0.2196423262357712 Conv_1_aggregated_features[9,7]
   + 0.0830722749233246 Conv_1_aggregated_features[9,8]
   - 0.4822347462177277 Conv_1_aggregated_features[9,9]
   + 0.5108910202980042 Conv_1_aggregated_features[9,10]
   + 3.214276790618896 Conv_1_aggregated_features[9,11]
   - 0.7224436402320862 Conv_1_aggregated_features[9,12]
   + 0.2678262591362 Conv_1_aggregated_features[9,13]
   - 0.8807565569877625 Conv_1_aggregated_features[9,14]
   - 0.3268090188503265 Conv_1_aggregated_features[9,15] + Conv_1_t[9,9]
   = -0.8283404111862183
 Conv_1_output_constraint[9,10]: 0.3653882741928101 Conv_0_Relu_ts[9,0]
   - 0.8208209872245789 Conv_0_Relu_ts[9,1]
   + 0.3088222146034241 Conv_0_Relu_ts[9,2]
   + 0.1173163652420044 Conv_0_Relu_ts[9,3]
   + 0.658578634262085 Conv_0_Relu_ts[9,4]
   + 0.584019660949707 Conv_0_Relu_ts[9,5]
   - 0.8654046058654785 Conv_0_Relu_ts[9,6]
   + 0.3697350919246674 Conv_0_Relu_ts[9,7]
   - 0.2274059355258942 Conv_0_Relu_ts[9,8]
   + 1.197938442230225 Conv_0_Relu_ts[9,9]
   + 0.2446341812610626 Conv_0_Relu_ts[9,10]
   - 0.5676743984222412 Conv_0_Relu_ts[9,11]
   - 0.3123646676540375 Conv_0_Relu_ts[9,12]
   + 3.181503534317017 Conv_0_Relu_ts[9,13]
   + 0.6926240921020508 Conv_0_Relu_ts[9,14]
   + 0.0886927545070648 Conv_0_Relu_ts[9,15]
   - 0.2005636394023895 Conv_1_aggregated_features[9,0]
   + 1.403087973594666 Conv_1_aggregated_features[9,1]
   + 0.0523448958992958 Conv_1_aggregated_features[9,2]
   + 0.2411543428897858 Conv_1_aggregated_features[9,3]
   + 0.5953171253204346 Conv_1_aggregated_features[9,4]
   - 0.8278964757919312 Conv_1_aggregated_features[9,5]
   + 0.2499217540025711 Conv_1_aggregated_features[9,6]
   - 0.7393644452095032 Conv_1_aggregated_features[9,7]
   + 0.0902875959873199 Conv_1_aggregated_features[9,8]
   + 0.1488711386919022 Conv_1_aggregated_features[9,9]
   - 0.0303693022578955 Conv_1_aggregated_features[9,10]
   + 2.024244070053101 Conv_1_aggregated_features[9,11]
   + 0.495431512594223 Conv_1_aggregated_features[9,12]
   + 0.3463623225688934 Conv_1_aggregated_features[9,13]
   - 0.0763808861374855 Conv_1_aggregated_features[9,14]
   + 0.1597562879323959 Conv_1_aggregated_features[9,15] + Conv_1_t[9,10]
   = -0.5063248872756958
 Conv_1_output_constraint[9,11]: - 0.5185666680335999 Conv_0_Relu_ts[9,0]
   + 3.656918525695801 Conv_0_Relu_ts[9,1]
   - 0.7082476615905762 Conv_0_Relu_ts[9,2]
   + 0.1922211349010468 Conv_0_Relu_ts[9,3]
   - 0.7149283289909363 Conv_0_Relu_ts[9,4]
   - 3.255658626556396 Conv_0_Relu_ts[9,5]
   + 2.464392185211182 Conv_0_Relu_ts[9,6]
   - 0.9477378129959106 Conv_0_Relu_ts[9,7]
   + 0.168237030506134 Conv_0_Relu_ts[9,8]
   - 1.177640557289124 Conv_0_Relu_ts[9,9]
   - 0.7062549591064453 Conv_0_Relu_ts[9,10]
   + 1.453806400299072 Conv_0_Relu_ts[9,11]
   + 0.2612711191177368 Conv_0_Relu_ts[9,12]
   + 0.2999382615089417 Conv_0_Relu_ts[9,13]
   - 1.255579590797424 Conv_0_Relu_ts[9,14]
   - 0.4265820384025574 Conv_0_Relu_ts[9,15]
   + 0.3379200994968414 Conv_1_aggregated_features[9,0]
   - 1.33229672908783 Conv_1_aggregated_features[9,1]
   - 0.1594242304563522 Conv_1_aggregated_features[9,2]
   + 0.2298935651779175 Conv_1_aggregated_features[9,3]
   - 1.311494588851929 Conv_1_aggregated_features[9,4]
   + 3.605161666870117 Conv_1_aggregated_features[9,5]
   - 0.3176752626895905 Conv_1_aggregated_features[9,6]
   + 0.3926282525062561 Conv_1_aggregated_features[9,7]
   + 0.1479305028915405 Conv_1_aggregated_features[9,8]
   - 0.0372937023639679 Conv_1_aggregated_features[9,9]
   + 0.3455427587032318 Conv_1_aggregated_features[9,10]
   - 2.002299070358276 Conv_1_aggregated_features[9,11]
   - 0.9904713034629822 Conv_1_aggregated_features[9,12]
   + 0.0751789212226868 Conv_1_aggregated_features[9,13]
   - 0.5166233777999878 Conv_1_aggregated_features[9,14]
   + 0.0510688796639442 Conv_1_aggregated_features[9,15] + Conv_1_t[9,11]
   = -1.480785727500916
 Conv_1_output_constraint[9,12]: 0.3080185949802399 Conv_0_Relu_ts[9,0]
   + 0.0994751453399658 Conv_0_Relu_ts[9,1]
   + 0.4488444924354553 Conv_0_Relu_ts[9,2]
   + 0.0305773317813873 Conv_0_Relu_ts[9,3]
   - 1.673393130302429 Conv_0_Relu_ts[9,4]
   + 0.2824500501155853 Conv_0_Relu_ts[9,5]
   - 1.192709684371948 Conv_0_Relu_ts[9,6]
   + 1.454243063926697 Conv_0_Relu_ts[9,7]
   + 0.1514801830053329 Conv_0_Relu_ts[9,8]
   + 1.026351928710938 Conv_0_Relu_ts[9,9]
   + 0.1183762699365616 Conv_0_Relu_ts[9,10]
   + 0.352224588394165 Conv_0_Relu_ts[9,11]
   + 1.090276837348938 Conv_0_Relu_ts[9,12]
   - 2.053535461425781 Conv_0_Relu_ts[9,13]
   + 2.538641452789307 Conv_0_Relu_ts[9,14]
   + 0.6387560963630676 Conv_0_Relu_ts[9,15]
   - 0.1990496516227722 Conv_1_aggregated_features[9,0]
   + 1.247365713119507 Conv_1_aggregated_features[9,1]
   + 0.2213435918092728 Conv_1_aggregated_features[9,2]
   + 0.2398849725723267 Conv_1_aggregated_features[9,3]
   - 2.001479864120483 Conv_1_aggregated_features[9,4]
   - 1.084800243377686 Conv_1_aggregated_features[9,5]
   + 0.4287458956241608 Conv_1_aggregated_features[9,6]
   - 0.8179780840873718 Conv_1_aggregated_features[9,7]
   - 0.0883614718914032 Conv_1_aggregated_features[9,8]
   - 0.4398066997528076 Conv_1_aggregated_features[9,9]
   + 0.2760448455810547 Conv_1_aggregated_features[9,10]
   + 1.944519639015198 Conv_1_aggregated_features[9,11]
   + 1.041437149047852 Conv_1_aggregated_features[9,12]
   - 1.682050704956055 Conv_1_aggregated_features[9,13]
   - 0.1825132220983505 Conv_1_aggregated_features[9,14]
   + 0.2881160080432892 Conv_1_aggregated_features[9,15] + Conv_1_t[9,12]
   = 2.457676649093628
 Conv_1_output_constraint[9,13]: 0.2853958606719971 Conv_0_Relu_ts[9,0]
   + 0.0445350483059883 Conv_0_Relu_ts[9,1]
   - 0.0597315393388271 Conv_0_Relu_ts[9,2]
   + 0.2403226792812347 Conv_0_Relu_ts[9,3]
   + 0.5103238224983215 Conv_0_Relu_ts[9,4]
   - 0.1369927227497101 Conv_0_Relu_ts[9,5]
   + 0.5091087818145752 Conv_0_Relu_ts[9,6]
   - 0.0576926171779633 Conv_0_Relu_ts[9,7]
   - 0.00701320171356201 Conv_0_Relu_ts[9,8]
   + 0.5782948136329651 Conv_0_Relu_ts[9,9]
   - 0.1188536584377289 Conv_0_Relu_ts[9,10]
   - 0.078026682138443 Conv_0_Relu_ts[9,11]
   + 0.1965750753879547 Conv_0_Relu_ts[9,12]
   + 0.4982857704162598 Conv_0_Relu_ts[9,13]
   - 0.0995257869362831 Conv_0_Relu_ts[9,14]
   + 0.0712818428874016 Conv_0_Relu_ts[9,15]
   + 0.2628078460693359 Conv_1_aggregated_features[9,0]
   + 0.6577805280685425 Conv_1_aggregated_features[9,1]
   - 0.1415485441684723 Conv_1_aggregated_features[9,2]
   + 0.1597791016101837 Conv_1_aggregated_features[9,3]
   + 0.3969745337963104 Conv_1_aggregated_features[9,4]
   - 0.3941676616668701 Conv_1_aggregated_features[9,5]
   + 0.3484204709529877 Conv_1_aggregated_features[9,6]
   + 0.2601790428161621 Conv_1_aggregated_features[9,7]
   - 0.0652478933334351 Conv_1_aggregated_features[9,8]
   + 0.3492280542850494 Conv_1_aggregated_features[9,9]
   + 0.1917867064476013 Conv_1_aggregated_features[9,10]
   + 0.2439181953668594 Conv_1_aggregated_features[9,11]
   + 0.1009163707494736 Conv_1_aggregated_features[9,12]
   + 0.2350947707891464 Conv_1_aggregated_features[9,13]
   + 0.2627962231636047 Conv_1_aggregated_features[9,14]
   - 0.0959920063614845 Conv_1_aggregated_features[9,15] + Conv_1_t[9,13]
   = -0.4268372058868408
 Conv_1_output_constraint[9,14]: - 0.2448655962944031 Conv_0_Relu_ts[9,0]
   + 0.2688764035701752 Conv_0_Relu_ts[9,1]
   + 0.0118266940116882 Conv_0_Relu_ts[9,2]
   - 0.0670525431632996 Conv_0_Relu_ts[9,3]
   + 0.1511579602956772 Conv_0_Relu_ts[9,4]
   - 0.4796198606491089 Conv_0_Relu_ts[9,5]
   - 0.1699391901493073 Conv_0_Relu_ts[9,6]
   + 0.2647148370742798 Conv_0_Relu_ts[9,7]
   + 0.137117475271225 Conv_0_Relu_ts[9,8]
   + 0.1737731248140335 Conv_0_Relu_ts[9,9]
   - 0.1610838174819946 Conv_0_Relu_ts[9,10]
   - 0.6311220526695251 Conv_0_Relu_ts[9,11]
   + 0.6263757348060608 Conv_0_Relu_ts[9,12]
   - 0.4823384284973145 Conv_0_Relu_ts[9,13]
   + 0.2192765176296234 Conv_0_Relu_ts[9,14]
   + 0.1739131808280945 Conv_0_Relu_ts[9,15]
   + 0.0443203188478947 Conv_1_aggregated_features[9,0]
   + 0.7668815851211548 Conv_1_aggregated_features[9,1]
   + 0.248365581035614 Conv_1_aggregated_features[9,2]
   - 0.2433404624462128 Conv_1_aggregated_features[9,3]
   + 0.2770798802375793 Conv_1_aggregated_features[9,4]
   - 1.224791765213013 Conv_1_aggregated_features[9,5]
   - 0.1892879456281662 Conv_1_aggregated_features[9,6]
   + 0.1141589805483818 Conv_1_aggregated_features[9,7]
   + 0.2022489905357361 Conv_1_aggregated_features[9,8]
   - 0.0391396470367908 Conv_1_aggregated_features[9,9]
   - 0.0935317650437355 Conv_1_aggregated_features[9,10]
   - 0.5407891273498535 Conv_1_aggregated_features[9,11]
   + 0.3130630552768707 Conv_1_aggregated_features[9,12]
   + 0.3410882651805878 Conv_1_aggregated_features[9,13]
   - 1.855417490005493 Conv_1_aggregated_features[9,14]
   + 0.247613787651062 Conv_1_aggregated_features[9,15] + Conv_1_t[9,14]
   = -0.2306040376424789
 Conv_1_output_constraint[9,15]: 0.043729841709137 Conv_0_Relu_ts[9,0]
   + 0.045797374099493 Conv_0_Relu_ts[9,1]
   - 0.1762047111988068 Conv_0_Relu_ts[9,2]
   - 0.2225021123886108 Conv_0_Relu_ts[9,3]
   + 0.1344186365604401 Conv_0_Relu_ts[9,4]
   + 0.2975046932697296 Conv_0_Relu_ts[9,5]
   + 0.1135610193014145 Conv_0_Relu_ts[9,6]
   + 0.165085032582283 Conv_0_Relu_ts[9,7]
   + 0.0632660016417503 Conv_0_Relu_ts[9,8]
   + 0.2534148395061493 Conv_0_Relu_ts[9,9]
   + 0.1100017726421356 Conv_0_Relu_ts[9,10]
   + 0.045901358127594 Conv_0_Relu_ts[9,11]
   + 0.2102706432342529 Conv_0_Relu_ts[9,12]
   + 0.1767551600933075 Conv_0_Relu_ts[9,13]
   - 0.0864512398838997 Conv_0_Relu_ts[9,14]
   + 0.1616824418306351 Conv_0_Relu_ts[9,15]
   + 0.0113421082496643 Conv_1_aggregated_features[9,0]
   + 0.2387615889310837 Conv_1_aggregated_features[9,1]
   + 0.1763307154178619 Conv_1_aggregated_features[9,2]
   + 0.1442219316959381 Conv_1_aggregated_features[9,3]
   + 0.3573774099349976 Conv_1_aggregated_features[9,4]
   - 0.0406427308917046 Conv_1_aggregated_features[9,5]
   + 0.1155626475811005 Conv_1_aggregated_features[9,6]
   + 0.0186243671923876 Conv_1_aggregated_features[9,7]
   - 0.1389756500720978 Conv_1_aggregated_features[9,8]
   + 0.274962306022644 Conv_1_aggregated_features[9,9]
   + 0.0309249386191368 Conv_1_aggregated_features[9,10]
   - 0.1606657803058624 Conv_1_aggregated_features[9,11]
   + 0.2456246316432953 Conv_1_aggregated_features[9,12]
   + 0.4163841009140015 Conv_1_aggregated_features[9,13]
   + 0.29454505443573 Conv_1_aggregated_features[9,14]
   + 0.1688967496156693 Conv_1_aggregated_features[9,15] + Conv_1_t[9,15]
   = -0.112743653357029
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
 Conv_1_Relu_constraint_1[8,0]: - Conv_1_t[8,0] + Conv_1_Relu_ts[8,0]
   - Conv_1_Relu_ss[8,0] = 0
 Conv_1_Relu_constraint_1[8,1]: - Conv_1_t[8,1] + Conv_1_Relu_ts[8,1]
   - Conv_1_Relu_ss[8,1] = 0
 Conv_1_Relu_constraint_1[8,2]: - Conv_1_t[8,2] + Conv_1_Relu_ts[8,2]
   - Conv_1_Relu_ss[8,2] = 0
 Conv_1_Relu_constraint_1[8,3]: - Conv_1_t[8,3] + Conv_1_Relu_ts[8,3]
   - Conv_1_Relu_ss[8,3] = 0
 Conv_1_Relu_constraint_1[8,4]: - Conv_1_t[8,4] + Conv_1_Relu_ts[8,4]
   - Conv_1_Relu_ss[8,4] = 0
 Conv_1_Relu_constraint_1[8,5]: - Conv_1_t[8,5] + Conv_1_Relu_ts[8,5]
   - Conv_1_Relu_ss[8,5] = 0
 Conv_1_Relu_constraint_1[8,6]: - Conv_1_t[8,6] + Conv_1_Relu_ts[8,6]
   - Conv_1_Relu_ss[8,6] = 0
 Conv_1_Relu_constraint_1[8,7]: - Conv_1_t[8,7] + Conv_1_Relu_ts[8,7]
   - Conv_1_Relu_ss[8,7] = 0
 Conv_1_Relu_constraint_1[8,8]: - Conv_1_t[8,8] + Conv_1_Relu_ts[8,8]
   - Conv_1_Relu_ss[8,8] = 0
 Conv_1_Relu_constraint_1[8,9]: - Conv_1_t[8,9] + Conv_1_Relu_ts[8,9]
   - Conv_1_Relu_ss[8,9] = 0
 Conv_1_Relu_constraint_1[8,10]: - Conv_1_t[8,10] + Conv_1_Relu_ts[8,10]
   - Conv_1_Relu_ss[8,10] = 0
 Conv_1_Relu_constraint_1[8,11]: - Conv_1_t[8,11] + Conv_1_Relu_ts[8,11]
   - Conv_1_Relu_ss[8,11] = 0
 Conv_1_Relu_constraint_1[8,12]: - Conv_1_t[8,12] + Conv_1_Relu_ts[8,12]
   - Conv_1_Relu_ss[8,12] = 0
 Conv_1_Relu_constraint_1[8,13]: - Conv_1_t[8,13] + Conv_1_Relu_ts[8,13]
   - Conv_1_Relu_ss[8,13] = 0
 Conv_1_Relu_constraint_1[8,14]: - Conv_1_t[8,14] + Conv_1_Relu_ts[8,14]
   - Conv_1_Relu_ss[8,14] = 0
 Conv_1_Relu_constraint_1[8,15]: - Conv_1_t[8,15] + Conv_1_Relu_ts[8,15]
   - Conv_1_Relu_ss[8,15] = 0
 Conv_1_Relu_constraint_1[9,0]: - Conv_1_t[9,0] + Conv_1_Relu_ts[9,0]
   - Conv_1_Relu_ss[9,0] = 0
 Conv_1_Relu_constraint_1[9,1]: - Conv_1_t[9,1] + Conv_1_Relu_ts[9,1]
   - Conv_1_Relu_ss[9,1] = 0
 Conv_1_Relu_constraint_1[9,2]: - Conv_1_t[9,2] + Conv_1_Relu_ts[9,2]
   - Conv_1_Relu_ss[9,2] = 0
 Conv_1_Relu_constraint_1[9,3]: - Conv_1_t[9,3] + Conv_1_Relu_ts[9,3]
   - Conv_1_Relu_ss[9,3] = 0
 Conv_1_Relu_constraint_1[9,4]: - Conv_1_t[9,4] + Conv_1_Relu_ts[9,4]
   - Conv_1_Relu_ss[9,4] = 0
 Conv_1_Relu_constraint_1[9,5]: - Conv_1_t[9,5] + Conv_1_Relu_ts[9,5]
   - Conv_1_Relu_ss[9,5] = 0
 Conv_1_Relu_constraint_1[9,6]: - Conv_1_t[9,6] + Conv_1_Relu_ts[9,6]
   - Conv_1_Relu_ss[9,6] = 0
 Conv_1_Relu_constraint_1[9,7]: - Conv_1_t[9,7] + Conv_1_Relu_ts[9,7]
   - Conv_1_Relu_ss[9,7] = 0
 Conv_1_Relu_constraint_1[9,8]: - Conv_1_t[9,8] + Conv_1_Relu_ts[9,8]
   - Conv_1_Relu_ss[9,8] = 0
 Conv_1_Relu_constraint_1[9,9]: - Conv_1_t[9,9] + Conv_1_Relu_ts[9,9]
   - Conv_1_Relu_ss[9,9] = 0
 Conv_1_Relu_constraint_1[9,10]: - Conv_1_t[9,10] + Conv_1_Relu_ts[9,10]
   - Conv_1_Relu_ss[9,10] = 0
 Conv_1_Relu_constraint_1[9,11]: - Conv_1_t[9,11] + Conv_1_Relu_ts[9,11]
   - Conv_1_Relu_ss[9,11] = 0
 Conv_1_Relu_constraint_1[9,12]: - Conv_1_t[9,12] + Conv_1_Relu_ts[9,12]
   - Conv_1_Relu_ss[9,12] = 0
 Conv_1_Relu_constraint_1[9,13]: - Conv_1_t[9,13] + Conv_1_Relu_ts[9,13]
   - Conv_1_Relu_ss[9,13] = 0
 Conv_1_Relu_constraint_1[9,14]: - Conv_1_t[9,14] + Conv_1_Relu_ts[9,14]
   - Conv_1_Relu_ss[9,14] = 0
 Conv_1_Relu_constraint_1[9,15]: - Conv_1_t[9,15] + Conv_1_Relu_ts[9,15]
   - Conv_1_Relu_ss[9,15] = 0
 Conv_1_Relu_constraint_2[0,0]: Conv_1_Relu_ts[0,0]
   - 25.57325229459289 C1400 <= 0
 Conv_1_Relu_constraint_2[0,1]: Conv_1_Relu_ts[0,1]
   - 16.63501010363109 C1401 <= 0
 Conv_1_Relu_constraint_2[0,2]: Conv_1_Relu_ts[0,2]
   - 35.00484918336429 C1402 <= 0
 Conv_1_Relu_constraint_2[0,3]: Conv_1_Relu_ts[0,3]
   - 0.4998085927964429 C1403 <= 0
 Conv_1_Relu_constraint_2[0,4]: Conv_1_Relu_ts[0,4]
   - 57.82216539781905 C1404 <= 0
 Conv_1_Relu_constraint_2[0,5]: Conv_1_Relu_ts[0,5]
   - 2.411270213327107 C1405 <= 0
 Conv_1_Relu_constraint_2[0,6]: Conv_1_Relu_ts[0,6]
   - 49.46156674751891 C1406 <= 0
 Conv_1_Relu_constraint_2[0,7]: Conv_1_Relu_ts[0,7]
   - 3.462243420878994 C1407 <= 0
 Conv_1_Relu_constraint_2[0,8]: Conv_1_Relu_ts[0,8]
   - 36.11140265688224 C1408 <= 0
 Conv_1_Relu_constraint_2[0,9]: Conv_1_Relu_ts[0,9]
   - 9.030128828439624 C1409 <= 0
 Conv_1_Relu_constraint_2[0,10]: Conv_1_Relu_ts[0,10]
   - 23.63937264816158 C1410 <= 0
 Conv_1_Relu_constraint_2[0,11]: Conv_1_Relu_ts[0,11]
   - 60.18996818642027 C1411 <= 0
 Conv_1_Relu_constraint_2[0,12]: Conv_1_Relu_ts[0,12]
   - 29.32487877957098 C1412 <= 0
 Conv_1_Relu_constraint_2[0,13]: Conv_1_Relu_ts[0,13]
   - 3.004389269438119 C1413 <= 0
 Conv_1_Relu_constraint_2[0,14]: Conv_1_Relu_ts[0,14]
   - 24.37010483280744 C1414 <= 0
 Conv_1_Relu_constraint_2[0,15]: Conv_1_Relu_ts[0,15]
   - 1.234436565832815 C1415 <= 0
 Conv_1_Relu_constraint_2[1,0]: Conv_1_Relu_ts[1,0]
   - 25.57325229459289 C1416 <= 0
 Conv_1_Relu_constraint_2[1,1]: Conv_1_Relu_ts[1,1]
   - 16.63501010363109 C1417 <= 0
 Conv_1_Relu_constraint_2[1,2]: Conv_1_Relu_ts[1,2]
   - 35.00484918336429 C1418 <= 0
 Conv_1_Relu_constraint_2[1,3]: Conv_1_Relu_ts[1,3]
   - 0.4998085927964429 C1419 <= 0
 Conv_1_Relu_constraint_2[1,4]: Conv_1_Relu_ts[1,4]
   - 57.82216539781905 C1420 <= 0
 Conv_1_Relu_constraint_2[1,5]: Conv_1_Relu_ts[1,5]
   - 2.411270213327107 C1421 <= 0
 Conv_1_Relu_constraint_2[1,6]: Conv_1_Relu_ts[1,6]
   - 49.46156674751891 C1422 <= 0
 Conv_1_Relu_constraint_2[1,7]: Conv_1_Relu_ts[1,7]
   - 3.462243420878994 C1423 <= 0
 Conv_1_Relu_constraint_2[1,8]: Conv_1_Relu_ts[1,8]
   - 36.11140265688224 C1424 <= 0
 Conv_1_Relu_constraint_2[1,9]: Conv_1_Relu_ts[1,9]
   - 9.030128828439624 C1425 <= 0
 Conv_1_Relu_constraint_2[1,10]: Conv_1_Relu_ts[1,10]
   - 23.63937264816158 C1426 <= 0
 Conv_1_Relu_constraint_2[1,11]: Conv_1_Relu_ts[1,11]
   - 60.18996818642027 C1427 <= 0
 Conv_1_Relu_constraint_2[1,12]: Conv_1_Relu_ts[1,12]
   - 29.32487877957098 C1428 <= 0
 Conv_1_Relu_constraint_2[1,13]: Conv_1_Relu_ts[1,13]
   - 3.004389269438119 C1429 <= 0
 Conv_1_Relu_constraint_2[1,14]: Conv_1_Relu_ts[1,14]
   - 24.37010483280744 C1430 <= 0
 Conv_1_Relu_constraint_2[1,15]: Conv_1_Relu_ts[1,15]
   - 1.234436565832815 C1431 <= 0
 Conv_1_Relu_constraint_2[2,0]: Conv_1_Relu_ts[2,0]
   - 25.57325229459289 C1432 <= 0
 Conv_1_Relu_constraint_2[2,1]: Conv_1_Relu_ts[2,1]
   - 16.63501010363109 C1433 <= 0
 Conv_1_Relu_constraint_2[2,2]: Conv_1_Relu_ts[2,2]
   - 35.00484918336429 C1434 <= 0
 Conv_1_Relu_constraint_2[2,3]: Conv_1_Relu_ts[2,3]
   - 0.4998085927964429 C1435 <= 0
 Conv_1_Relu_constraint_2[2,4]: Conv_1_Relu_ts[2,4]
   - 57.82216539781905 C1436 <= 0
 Conv_1_Relu_constraint_2[2,5]: Conv_1_Relu_ts[2,5]
   - 2.411270213327107 C1437 <= 0
 Conv_1_Relu_constraint_2[2,6]: Conv_1_Relu_ts[2,6]
   - 49.46156674751891 C1438 <= 0
 Conv_1_Relu_constraint_2[2,7]: Conv_1_Relu_ts[2,7]
   - 3.462243420878994 C1439 <= 0
 Conv_1_Relu_constraint_2[2,8]: Conv_1_Relu_ts[2,8]
   - 36.11140265688224 C1440 <= 0
 Conv_1_Relu_constraint_2[2,9]: Conv_1_Relu_ts[2,9]
   - 9.030128828439624 C1441 <= 0
 Conv_1_Relu_constraint_2[2,10]: Conv_1_Relu_ts[2,10]
   - 23.63937264816158 C1442 <= 0
 Conv_1_Relu_constraint_2[2,11]: Conv_1_Relu_ts[2,11]
   - 60.18996818642027 C1443 <= 0
 Conv_1_Relu_constraint_2[2,12]: Conv_1_Relu_ts[2,12]
   - 29.32487877957098 C1444 <= 0
 Conv_1_Relu_constraint_2[2,13]: Conv_1_Relu_ts[2,13]
   - 3.004389269438119 C1445 <= 0
 Conv_1_Relu_constraint_2[2,14]: Conv_1_Relu_ts[2,14]
   - 24.37010483280744 C1446 <= 0
 Conv_1_Relu_constraint_2[2,15]: Conv_1_Relu_ts[2,15]
   - 1.234436565832815 C1447 <= 0
 Conv_1_Relu_constraint_2[3,0]: Conv_1_Relu_ts[3,0]
   - 25.57325229459289 C1448 <= 0
 Conv_1_Relu_constraint_2[3,1]: Conv_1_Relu_ts[3,1]
   - 16.63501010363109 C1449 <= 0
 Conv_1_Relu_constraint_2[3,2]: Conv_1_Relu_ts[3,2]
   - 35.00484918336429 C1450 <= 0
 Conv_1_Relu_constraint_2[3,3]: Conv_1_Relu_ts[3,3]
   - 0.4998085927964429 C1451 <= 0
 Conv_1_Relu_constraint_2[3,4]: Conv_1_Relu_ts[3,4]
   - 57.82216539781905 C1452 <= 0
 Conv_1_Relu_constraint_2[3,5]: Conv_1_Relu_ts[3,5]
   - 2.411270213327107 C1453 <= 0
 Conv_1_Relu_constraint_2[3,6]: Conv_1_Relu_ts[3,6]
   - 49.46156674751891 C1454 <= 0
 Conv_1_Relu_constraint_2[3,7]: Conv_1_Relu_ts[3,7]
   - 3.462243420878994 C1455 <= 0
 Conv_1_Relu_constraint_2[3,8]: Conv_1_Relu_ts[3,8]
   - 36.11140265688224 C1456 <= 0
 Conv_1_Relu_constraint_2[3,9]: Conv_1_Relu_ts[3,9]
   - 9.030128828439624 C1457 <= 0
 Conv_1_Relu_constraint_2[3,10]: Conv_1_Relu_ts[3,10]
   - 23.63937264816158 C1458 <= 0
 Conv_1_Relu_constraint_2[3,11]: Conv_1_Relu_ts[3,11]
   - 60.18996818642027 C1459 <= 0
 Conv_1_Relu_constraint_2[3,12]: Conv_1_Relu_ts[3,12]
   - 29.32487877957098 C1460 <= 0
 Conv_1_Relu_constraint_2[3,13]: Conv_1_Relu_ts[3,13]
   - 3.004389269438119 C1461 <= 0
 Conv_1_Relu_constraint_2[3,14]: Conv_1_Relu_ts[3,14]
   - 24.37010483280744 C1462 <= 0
 Conv_1_Relu_constraint_2[3,15]: Conv_1_Relu_ts[3,15]
   - 1.234436565832815 C1463 <= 0
 Conv_1_Relu_constraint_2[4,0]: Conv_1_Relu_ts[4,0]
   - 25.57325229459289 C1464 <= 0
 Conv_1_Relu_constraint_2[4,1]: Conv_1_Relu_ts[4,1]
   - 16.63501010363109 C1465 <= 0
 Conv_1_Relu_constraint_2[4,2]: Conv_1_Relu_ts[4,2]
   - 35.00484918336429 C1466 <= 0
 Conv_1_Relu_constraint_2[4,3]: Conv_1_Relu_ts[4,3]
   - 0.4998085927964429 C1467 <= 0
 Conv_1_Relu_constraint_2[4,4]: Conv_1_Relu_ts[4,4]
   - 57.82216539781905 C1468 <= 0
 Conv_1_Relu_constraint_2[4,5]: Conv_1_Relu_ts[4,5]
   - 2.411270213327107 C1469 <= 0
 Conv_1_Relu_constraint_2[4,6]: Conv_1_Relu_ts[4,6]
   - 49.46156674751891 C1470 <= 0
 Conv_1_Relu_constraint_2[4,7]: Conv_1_Relu_ts[4,7]
   - 3.462243420878994 C1471 <= 0
 Conv_1_Relu_constraint_2[4,8]: Conv_1_Relu_ts[4,8]
   - 36.11140265688224 C1472 <= 0
 Conv_1_Relu_constraint_2[4,9]: Conv_1_Relu_ts[4,9]
   - 9.030128828439624 C1473 <= 0
 Conv_1_Relu_constraint_2[4,10]: Conv_1_Relu_ts[4,10]
   - 23.63937264816158 C1474 <= 0
 Conv_1_Relu_constraint_2[4,11]: Conv_1_Relu_ts[4,11]
   - 60.18996818642027 C1475 <= 0
 Conv_1_Relu_constraint_2[4,12]: Conv_1_Relu_ts[4,12]
   - 29.32487877957098 C1476 <= 0
 Conv_1_Relu_constraint_2[4,13]: Conv_1_Relu_ts[4,13]
   - 3.004389269438119 C1477 <= 0
 Conv_1_Relu_constraint_2[4,14]: Conv_1_Relu_ts[4,14]
   - 24.37010483280744 C1478 <= 0
 Conv_1_Relu_constraint_2[4,15]: Conv_1_Relu_ts[4,15]
   - 1.234436565832815 C1479 <= 0
 Conv_1_Relu_constraint_2[5,0]: Conv_1_Relu_ts[5,0]
   - 25.57325229459289 C1480 <= 0
 Conv_1_Relu_constraint_2[5,1]: Conv_1_Relu_ts[5,1]
   - 16.63501010363109 C1481 <= 0
 Conv_1_Relu_constraint_2[5,2]: Conv_1_Relu_ts[5,2]
   - 35.00484918336429 C1482 <= 0
 Conv_1_Relu_constraint_2[5,3]: Conv_1_Relu_ts[5,3]
   - 0.4998085927964429 C1483 <= 0
 Conv_1_Relu_constraint_2[5,4]: Conv_1_Relu_ts[5,4]
   - 57.82216539781905 C1484 <= 0
 Conv_1_Relu_constraint_2[5,5]: Conv_1_Relu_ts[5,5]
   - 2.411270213327107 C1485 <= 0
 Conv_1_Relu_constraint_2[5,6]: Conv_1_Relu_ts[5,6]
   - 49.46156674751891 C1486 <= 0
 Conv_1_Relu_constraint_2[5,7]: Conv_1_Relu_ts[5,7]
   - 3.462243420878994 C1487 <= 0
 Conv_1_Relu_constraint_2[5,8]: Conv_1_Relu_ts[5,8]
   - 36.11140265688224 C1488 <= 0
 Conv_1_Relu_constraint_2[5,9]: Conv_1_Relu_ts[5,9]
   - 9.030128828439624 C1489 <= 0
 Conv_1_Relu_constraint_2[5,10]: Conv_1_Relu_ts[5,10]
   - 23.63937264816158 C1490 <= 0
 Conv_1_Relu_constraint_2[5,11]: Conv_1_Relu_ts[5,11]
   - 60.18996818642027 C1491 <= 0
 Conv_1_Relu_constraint_2[5,12]: Conv_1_Relu_ts[5,12]
   - 29.32487877957098 C1492 <= 0
 Conv_1_Relu_constraint_2[5,13]: Conv_1_Relu_ts[5,13]
   - 3.004389269438119 C1493 <= 0
 Conv_1_Relu_constraint_2[5,14]: Conv_1_Relu_ts[5,14]
   - 24.37010483280744 C1494 <= 0
 Conv_1_Relu_constraint_2[5,15]: Conv_1_Relu_ts[5,15]
   - 1.234436565832815 C1495 <= 0
 Conv_1_Relu_constraint_2[6,0]: Conv_1_Relu_ts[6,0]
   - 25.57325229459289 C1496 <= 0
 Conv_1_Relu_constraint_2[6,1]: Conv_1_Relu_ts[6,1]
   - 16.63501010363109 C1497 <= 0
 Conv_1_Relu_constraint_2[6,2]: Conv_1_Relu_ts[6,2]
   - 35.00484918336429 C1498 <= 0
 Conv_1_Relu_constraint_2[6,3]: Conv_1_Relu_ts[6,3]
   - 0.4998085927964429 C1499 <= 0
 Conv_1_Relu_constraint_2[6,4]: Conv_1_Relu_ts[6,4]
   - 57.82216539781905 C1500 <= 0
 Conv_1_Relu_constraint_2[6,5]: Conv_1_Relu_ts[6,5]
   - 2.411270213327107 C1501 <= 0
 Conv_1_Relu_constraint_2[6,6]: Conv_1_Relu_ts[6,6]
   - 49.46156674751891 C1502 <= 0
 Conv_1_Relu_constraint_2[6,7]: Conv_1_Relu_ts[6,7]
   - 3.462243420878994 C1503 <= 0
 Conv_1_Relu_constraint_2[6,8]: Conv_1_Relu_ts[6,8]
   - 36.11140265688224 C1504 <= 0
 Conv_1_Relu_constraint_2[6,9]: Conv_1_Relu_ts[6,9]
   - 9.030128828439624 C1505 <= 0
 Conv_1_Relu_constraint_2[6,10]: Conv_1_Relu_ts[6,10]
   - 23.63937264816158 C1506 <= 0
 Conv_1_Relu_constraint_2[6,11]: Conv_1_Relu_ts[6,11]
   - 60.18996818642027 C1507 <= 0
 Conv_1_Relu_constraint_2[6,12]: Conv_1_Relu_ts[6,12]
   - 29.32487877957098 C1508 <= 0
 Conv_1_Relu_constraint_2[6,13]: Conv_1_Relu_ts[6,13]
   - 3.004389269438119 C1509 <= 0
 Conv_1_Relu_constraint_2[6,14]: Conv_1_Relu_ts[6,14]
   - 24.37010483280744 C1510 <= 0
 Conv_1_Relu_constraint_2[6,15]: Conv_1_Relu_ts[6,15]
   - 1.234436565832815 C1511 <= 0
 Conv_1_Relu_constraint_2[7,0]: Conv_1_Relu_ts[7,0]
   - 25.57325229459289 C1512 <= 0
 Conv_1_Relu_constraint_2[7,1]: Conv_1_Relu_ts[7,1]
   - 16.63501010363109 C1513 <= 0
 Conv_1_Relu_constraint_2[7,2]: Conv_1_Relu_ts[7,2]
   - 35.00484918336429 C1514 <= 0
 Conv_1_Relu_constraint_2[7,3]: Conv_1_Relu_ts[7,3]
   - 0.4998085927964429 C1515 <= 0
 Conv_1_Relu_constraint_2[7,4]: Conv_1_Relu_ts[7,4]
   - 57.82216539781905 C1516 <= 0
 Conv_1_Relu_constraint_2[7,5]: Conv_1_Relu_ts[7,5]
   - 2.411270213327107 C1517 <= 0
 Conv_1_Relu_constraint_2[7,6]: Conv_1_Relu_ts[7,6]
   - 49.46156674751891 C1518 <= 0
 Conv_1_Relu_constraint_2[7,7]: Conv_1_Relu_ts[7,7]
   - 3.462243420878994 C1519 <= 0
 Conv_1_Relu_constraint_2[7,8]: Conv_1_Relu_ts[7,8]
   - 36.11140265688224 C1520 <= 0
 Conv_1_Relu_constraint_2[7,9]: Conv_1_Relu_ts[7,9]
   - 9.030128828439624 C1521 <= 0
 Conv_1_Relu_constraint_2[7,10]: Conv_1_Relu_ts[7,10]
   - 23.63937264816158 C1522 <= 0
 Conv_1_Relu_constraint_2[7,11]: Conv_1_Relu_ts[7,11]
   - 60.18996818642027 C1523 <= 0
 Conv_1_Relu_constraint_2[7,12]: Conv_1_Relu_ts[7,12]
   - 29.32487877957098 C1524 <= 0
 Conv_1_Relu_constraint_2[7,13]: Conv_1_Relu_ts[7,13]
   - 3.004389269438119 C1525 <= 0
 Conv_1_Relu_constraint_2[7,14]: Conv_1_Relu_ts[7,14]
   - 24.37010483280744 C1526 <= 0
 Conv_1_Relu_constraint_2[7,15]: Conv_1_Relu_ts[7,15]
   - 1.234436565832815 C1527 <= 0
 Conv_1_Relu_constraint_2[8,0]: Conv_1_Relu_ts[8,0]
   - 25.57325229459289 C1528 <= 0
 Conv_1_Relu_constraint_2[8,1]: Conv_1_Relu_ts[8,1]
   - 16.63501010363109 C1529 <= 0
 Conv_1_Relu_constraint_2[8,2]: Conv_1_Relu_ts[8,2]
   - 35.00484918336429 C1530 <= 0
 Conv_1_Relu_constraint_2[8,3]: Conv_1_Relu_ts[8,3]
   - 0.4998085927964429 C1531 <= 0
 Conv_1_Relu_constraint_2[8,4]: Conv_1_Relu_ts[8,4]
   - 57.82216539781905 C1532 <= 0
 Conv_1_Relu_constraint_2[8,5]: Conv_1_Relu_ts[8,5]
   - 2.411270213327107 C1533 <= 0
 Conv_1_Relu_constraint_2[8,6]: Conv_1_Relu_ts[8,6]
   - 49.46156674751891 C1534 <= 0
 Conv_1_Relu_constraint_2[8,7]: Conv_1_Relu_ts[8,7]
   - 3.462243420878994 C1535 <= 0
 Conv_1_Relu_constraint_2[8,8]: Conv_1_Relu_ts[8,8]
   - 36.11140265688224 C1536 <= 0
 Conv_1_Relu_constraint_2[8,9]: Conv_1_Relu_ts[8,9]
   - 9.030128828439624 C1537 <= 0
 Conv_1_Relu_constraint_2[8,10]: Conv_1_Relu_ts[8,10]
   - 23.63937264816158 C1538 <= 0
 Conv_1_Relu_constraint_2[8,11]: Conv_1_Relu_ts[8,11]
   - 60.18996818642027 C1539 <= 0
 Conv_1_Relu_constraint_2[8,12]: Conv_1_Relu_ts[8,12]
   - 29.32487877957098 C1540 <= 0
 Conv_1_Relu_constraint_2[8,13]: Conv_1_Relu_ts[8,13]
   - 3.004389269438119 C1541 <= 0
 Conv_1_Relu_constraint_2[8,14]: Conv_1_Relu_ts[8,14]
   - 24.37010483280744 C1542 <= 0
 Conv_1_Relu_constraint_2[8,15]: Conv_1_Relu_ts[8,15]
   - 1.234436565832815 C1543 <= 0
 Conv_1_Relu_constraint_2[9,0]: Conv_1_Relu_ts[9,0]
   - 25.57325229459289 C1544 <= 0
 Conv_1_Relu_constraint_2[9,1]: Conv_1_Relu_ts[9,1]
   - 16.63501010363109 C1545 <= 0
 Conv_1_Relu_constraint_2[9,2]: Conv_1_Relu_ts[9,2]
   - 35.00484918336429 C1546 <= 0
 Conv_1_Relu_constraint_2[9,3]: Conv_1_Relu_ts[9,3]
   - 0.4998085927964429 C1547 <= 0
 Conv_1_Relu_constraint_2[9,4]: Conv_1_Relu_ts[9,4]
   - 57.82216539781905 C1548 <= 0
 Conv_1_Relu_constraint_2[9,5]: Conv_1_Relu_ts[9,5]
   - 2.411270213327107 C1549 <= 0
 Conv_1_Relu_constraint_2[9,6]: Conv_1_Relu_ts[9,6]
   - 49.46156674751891 C1550 <= 0
 Conv_1_Relu_constraint_2[9,7]: Conv_1_Relu_ts[9,7]
   - 3.462243420878994 C1551 <= 0
 Conv_1_Relu_constraint_2[9,8]: Conv_1_Relu_ts[9,8]
   - 36.11140265688224 C1552 <= 0
 Conv_1_Relu_constraint_2[9,9]: Conv_1_Relu_ts[9,9]
   - 9.030128828439624 C1553 <= 0
 Conv_1_Relu_constraint_2[9,10]: Conv_1_Relu_ts[9,10]
   - 23.63937264816158 C1554 <= 0
 Conv_1_Relu_constraint_2[9,11]: Conv_1_Relu_ts[9,11]
   - 60.18996818642027 C1555 <= 0
 Conv_1_Relu_constraint_2[9,12]: Conv_1_Relu_ts[9,12]
   - 29.32487877957098 C1556 <= 0
 Conv_1_Relu_constraint_2[9,13]: Conv_1_Relu_ts[9,13]
   - 3.004389269438119 C1557 <= 0
 Conv_1_Relu_constraint_2[9,14]: Conv_1_Relu_ts[9,14]
   - 24.37010483280744 C1558 <= 0
 Conv_1_Relu_constraint_2[9,15]: Conv_1_Relu_ts[9,15]
   - 1.234436565832815 C1559 <= 0
 Conv_1_Relu_constraint_3[0,0]: Conv_1_Relu_ss[0,0]
   + 56.29245011536772 C1400 <= 56.29245011536772
 Conv_1_Relu_constraint_3[0,1]: Conv_1_Relu_ss[0,1]
   + 67.52969714306539 C1401 <= 67.52969714306539
 Conv_1_Relu_constraint_3[0,2]: Conv_1_Relu_ss[0,2]
   + 88.95482652838119 C1402 <= 88.95482652838119
 Conv_1_Relu_constraint_3[0,3]: Conv_1_Relu_ss[0,3]
   + 33.94596106828702 C1403 <= 33.94596106828702
 Conv_1_Relu_constraint_3[0,4]: Conv_1_Relu_ss[0,4]
   + 106.1251358090966 C1404 <= 106.1251358090966
 Conv_1_Relu_constraint_3[0,5]: Conv_1_Relu_ss[0,5]
   + 10.77560081932031 C1405 <= 10.77560081932031
 Conv_1_Relu_constraint_3[0,6]: Conv_1_Relu_ss[0,6]
   + 29.32218285779035 C1406 <= 29.32218285779035
 Conv_1_Relu_constraint_3[0,7]: Conv_1_Relu_ss[0,7]
   + 24.30472028988515 C1407 <= 24.30472028988515
 Conv_1_Relu_constraint_3[0,8]: Conv_1_Relu_ss[0,8]
   + 25.8812298511837 C1408 <= 25.8812298511837
 Conv_1_Relu_constraint_3[0,9]: Conv_1_Relu_ss[0,9]
   + 76.75665981089216 C1409 <= 76.75665981089216
 Conv_1_Relu_constraint_3[0,10]: Conv_1_Relu_ss[0,10]
   + 54.2754182075725 C1410 <= 54.2754182075725
 Conv_1_Relu_constraint_3[0,11]: Conv_1_Relu_ss[0,11]
   + 97.18723355279275 C1411 <= 97.18723355279275
 Conv_1_Relu_constraint_3[0,12]: Conv_1_Relu_ss[0,12]
   + 47.62093859518428 C1412 <= 47.62093859518428
 Conv_1_Relu_constraint_3[0,13]: Conv_1_Relu_ss[0,13]
   + 21.10965662228852 C1413 <= 21.10965662228852
 Conv_1_Relu_constraint_3[0,14]: Conv_1_Relu_ss[0,14]
   + 21.61040402775194 C1414 <= 21.61040402775194
 Conv_1_Relu_constraint_3[0,15]: Conv_1_Relu_ss[0,15]
   + 11.46170062280965 C1415 <= 11.46170062280965
 Conv_1_Relu_constraint_3[1,0]: Conv_1_Relu_ss[1,0]
   + 56.29245011536772 C1416 <= 56.29245011536772
 Conv_1_Relu_constraint_3[1,1]: Conv_1_Relu_ss[1,1]
   + 67.52969714306539 C1417 <= 67.52969714306539
 Conv_1_Relu_constraint_3[1,2]: Conv_1_Relu_ss[1,2]
   + 88.95482652838119 C1418 <= 88.95482652838119
 Conv_1_Relu_constraint_3[1,3]: Conv_1_Relu_ss[1,3]
   + 33.94596106828702 C1419 <= 33.94596106828702
 Conv_1_Relu_constraint_3[1,4]: Conv_1_Relu_ss[1,4]
   + 106.1251358090966 C1420 <= 106.1251358090966
 Conv_1_Relu_constraint_3[1,5]: Conv_1_Relu_ss[1,5]
   + 10.77560081932031 C1421 <= 10.77560081932031
 Conv_1_Relu_constraint_3[1,6]: Conv_1_Relu_ss[1,6]
   + 29.32218285779035 C1422 <= 29.32218285779035
 Conv_1_Relu_constraint_3[1,7]: Conv_1_Relu_ss[1,7]
   + 24.30472028988515 C1423 <= 24.30472028988515
 Conv_1_Relu_constraint_3[1,8]: Conv_1_Relu_ss[1,8]
   + 25.8812298511837 C1424 <= 25.8812298511837
 Conv_1_Relu_constraint_3[1,9]: Conv_1_Relu_ss[1,9]
   + 76.75665981089216 C1425 <= 76.75665981089216
 Conv_1_Relu_constraint_3[1,10]: Conv_1_Relu_ss[1,10]
   + 54.2754182075725 C1426 <= 54.2754182075725
 Conv_1_Relu_constraint_3[1,11]: Conv_1_Relu_ss[1,11]
   + 97.18723355279275 C1427 <= 97.18723355279275
 Conv_1_Relu_constraint_3[1,12]: Conv_1_Relu_ss[1,12]
   + 47.62093859518428 C1428 <= 47.62093859518428
 Conv_1_Relu_constraint_3[1,13]: Conv_1_Relu_ss[1,13]
   + 21.10965662228852 C1429 <= 21.10965662228852
 Conv_1_Relu_constraint_3[1,14]: Conv_1_Relu_ss[1,14]
   + 21.61040402775194 C1430 <= 21.61040402775194
 Conv_1_Relu_constraint_3[1,15]: Conv_1_Relu_ss[1,15]
   + 11.46170062280965 C1431 <= 11.46170062280965
 Conv_1_Relu_constraint_3[2,0]: Conv_1_Relu_ss[2,0]
   + 56.29245011536772 C1432 <= 56.29245011536772
 Conv_1_Relu_constraint_3[2,1]: Conv_1_Relu_ss[2,1]
   + 67.52969714306539 C1433 <= 67.52969714306539
 Conv_1_Relu_constraint_3[2,2]: Conv_1_Relu_ss[2,2]
   + 88.95482652838119 C1434 <= 88.95482652838119
 Conv_1_Relu_constraint_3[2,3]: Conv_1_Relu_ss[2,3]
   + 33.94596106828702 C1435 <= 33.94596106828702
 Conv_1_Relu_constraint_3[2,4]: Conv_1_Relu_ss[2,4]
   + 106.1251358090966 C1436 <= 106.1251358090966
 Conv_1_Relu_constraint_3[2,5]: Conv_1_Relu_ss[2,5]
   + 10.77560081932031 C1437 <= 10.77560081932031
 Conv_1_Relu_constraint_3[2,6]: Conv_1_Relu_ss[2,6]
   + 29.32218285779035 C1438 <= 29.32218285779035
 Conv_1_Relu_constraint_3[2,7]: Conv_1_Relu_ss[2,7]
   + 24.30472028988515 C1439 <= 24.30472028988515
 Conv_1_Relu_constraint_3[2,8]: Conv_1_Relu_ss[2,8]
   + 25.8812298511837 C1440 <= 25.8812298511837
 Conv_1_Relu_constraint_3[2,9]: Conv_1_Relu_ss[2,9]
   + 76.75665981089216 C1441 <= 76.75665981089216
 Conv_1_Relu_constraint_3[2,10]: Conv_1_Relu_ss[2,10]
   + 54.2754182075725 C1442 <= 54.2754182075725
 Conv_1_Relu_constraint_3[2,11]: Conv_1_Relu_ss[2,11]
   + 97.18723355279275 C1443 <= 97.18723355279275
 Conv_1_Relu_constraint_3[2,12]: Conv_1_Relu_ss[2,12]
   + 47.62093859518428 C1444 <= 47.62093859518428
 Conv_1_Relu_constraint_3[2,13]: Conv_1_Relu_ss[2,13]
   + 21.10965662228852 C1445 <= 21.10965662228852
 Conv_1_Relu_constraint_3[2,14]: Conv_1_Relu_ss[2,14]
   + 21.61040402775194 C1446 <= 21.61040402775194
 Conv_1_Relu_constraint_3[2,15]: Conv_1_Relu_ss[2,15]
   + 11.46170062280965 C1447 <= 11.46170062280965
 Conv_1_Relu_constraint_3[3,0]: Conv_1_Relu_ss[3,0]
   + 56.29245011536772 C1448 <= 56.29245011536772
 Conv_1_Relu_constraint_3[3,1]: Conv_1_Relu_ss[3,1]
   + 67.52969714306539 C1449 <= 67.52969714306539
 Conv_1_Relu_constraint_3[3,2]: Conv_1_Relu_ss[3,2]
   + 88.95482652838119 C1450 <= 88.95482652838119
 Conv_1_Relu_constraint_3[3,3]: Conv_1_Relu_ss[3,3]
   + 33.94596106828702 C1451 <= 33.94596106828702
 Conv_1_Relu_constraint_3[3,4]: Conv_1_Relu_ss[3,4]
   + 106.1251358090966 C1452 <= 106.1251358090966
 Conv_1_Relu_constraint_3[3,5]: Conv_1_Relu_ss[3,5]
   + 10.77560081932031 C1453 <= 10.77560081932031
 Conv_1_Relu_constraint_3[3,6]: Conv_1_Relu_ss[3,6]
   + 29.32218285779035 C1454 <= 29.32218285779035
 Conv_1_Relu_constraint_3[3,7]: Conv_1_Relu_ss[3,7]
   + 24.30472028988515 C1455 <= 24.30472028988515
 Conv_1_Relu_constraint_3[3,8]: Conv_1_Relu_ss[3,8]
   + 25.8812298511837 C1456 <= 25.8812298511837
 Conv_1_Relu_constraint_3[3,9]: Conv_1_Relu_ss[3,9]
   + 76.75665981089216 C1457 <= 76.75665981089216
 Conv_1_Relu_constraint_3[3,10]: Conv_1_Relu_ss[3,10]
   + 54.2754182075725 C1458 <= 54.2754182075725
 Conv_1_Relu_constraint_3[3,11]: Conv_1_Relu_ss[3,11]
   + 97.18723355279275 C1459 <= 97.18723355279275
 Conv_1_Relu_constraint_3[3,12]: Conv_1_Relu_ss[3,12]
   + 47.62093859518428 C1460 <= 47.62093859518428
 Conv_1_Relu_constraint_3[3,13]: Conv_1_Relu_ss[3,13]
   + 21.10965662228852 C1461 <= 21.10965662228852
 Conv_1_Relu_constraint_3[3,14]: Conv_1_Relu_ss[3,14]
   + 21.61040402775194 C1462 <= 21.61040402775194
 Conv_1_Relu_constraint_3[3,15]: Conv_1_Relu_ss[3,15]
   + 11.46170062280965 C1463 <= 11.46170062280965
 Conv_1_Relu_constraint_3[4,0]: Conv_1_Relu_ss[4,0]
   + 56.29245011536772 C1464 <= 56.29245011536772
 Conv_1_Relu_constraint_3[4,1]: Conv_1_Relu_ss[4,1]
   + 67.52969714306539 C1465 <= 67.52969714306539
 Conv_1_Relu_constraint_3[4,2]: Conv_1_Relu_ss[4,2]
   + 88.95482652838119 C1466 <= 88.95482652838119
 Conv_1_Relu_constraint_3[4,3]: Conv_1_Relu_ss[4,3]
   + 33.94596106828702 C1467 <= 33.94596106828702
 Conv_1_Relu_constraint_3[4,4]: Conv_1_Relu_ss[4,4]
   + 106.1251358090966 C1468 <= 106.1251358090966
 Conv_1_Relu_constraint_3[4,5]: Conv_1_Relu_ss[4,5]
   + 10.77560081932031 C1469 <= 10.77560081932031
 Conv_1_Relu_constraint_3[4,6]: Conv_1_Relu_ss[4,6]
   + 29.32218285779035 C1470 <= 29.32218285779035
 Conv_1_Relu_constraint_3[4,7]: Conv_1_Relu_ss[4,7]
   + 24.30472028988515 C1471 <= 24.30472028988515
 Conv_1_Relu_constraint_3[4,8]: Conv_1_Relu_ss[4,8]
   + 25.8812298511837 C1472 <= 25.8812298511837
 Conv_1_Relu_constraint_3[4,9]: Conv_1_Relu_ss[4,9]
   + 76.75665981089216 C1473 <= 76.75665981089216
 Conv_1_Relu_constraint_3[4,10]: Conv_1_Relu_ss[4,10]
   + 54.2754182075725 C1474 <= 54.2754182075725
 Conv_1_Relu_constraint_3[4,11]: Conv_1_Relu_ss[4,11]
   + 97.18723355279275 C1475 <= 97.18723355279275
 Conv_1_Relu_constraint_3[4,12]: Conv_1_Relu_ss[4,12]
   + 47.62093859518428 C1476 <= 47.62093859518428
 Conv_1_Relu_constraint_3[4,13]: Conv_1_Relu_ss[4,13]
   + 21.10965662228852 C1477 <= 21.10965662228852
 Conv_1_Relu_constraint_3[4,14]: Conv_1_Relu_ss[4,14]
   + 21.61040402775194 C1478 <= 21.61040402775194
 Conv_1_Relu_constraint_3[4,15]: Conv_1_Relu_ss[4,15]
   + 11.46170062280965 C1479 <= 11.46170062280965
 Conv_1_Relu_constraint_3[5,0]: Conv_1_Relu_ss[5,0]
   + 56.29245011536772 C1480 <= 56.29245011536772
 Conv_1_Relu_constraint_3[5,1]: Conv_1_Relu_ss[5,1]
   + 67.52969714306539 C1481 <= 67.52969714306539
 Conv_1_Relu_constraint_3[5,2]: Conv_1_Relu_ss[5,2]
   + 88.95482652838119 C1482 <= 88.95482652838119
 Conv_1_Relu_constraint_3[5,3]: Conv_1_Relu_ss[5,3]
   + 33.94596106828702 C1483 <= 33.94596106828702
 Conv_1_Relu_constraint_3[5,4]: Conv_1_Relu_ss[5,4]
   + 106.1251358090966 C1484 <= 106.1251358090966
 Conv_1_Relu_constraint_3[5,5]: Conv_1_Relu_ss[5,5]
   + 10.77560081932031 C1485 <= 10.77560081932031
 Conv_1_Relu_constraint_3[5,6]: Conv_1_Relu_ss[5,6]
   + 29.32218285779035 C1486 <= 29.32218285779035
 Conv_1_Relu_constraint_3[5,7]: Conv_1_Relu_ss[5,7]
   + 24.30472028988515 C1487 <= 24.30472028988515
 Conv_1_Relu_constraint_3[5,8]: Conv_1_Relu_ss[5,8]
   + 25.8812298511837 C1488 <= 25.8812298511837
 Conv_1_Relu_constraint_3[5,9]: Conv_1_Relu_ss[5,9]
   + 76.75665981089216 C1489 <= 76.75665981089216
 Conv_1_Relu_constraint_3[5,10]: Conv_1_Relu_ss[5,10]
   + 54.2754182075725 C1490 <= 54.2754182075725
 Conv_1_Relu_constraint_3[5,11]: Conv_1_Relu_ss[5,11]
   + 97.18723355279275 C1491 <= 97.18723355279275
 Conv_1_Relu_constraint_3[5,12]: Conv_1_Relu_ss[5,12]
   + 47.62093859518428 C1492 <= 47.62093859518428
 Conv_1_Relu_constraint_3[5,13]: Conv_1_Relu_ss[5,13]
   + 21.10965662228852 C1493 <= 21.10965662228852
 Conv_1_Relu_constraint_3[5,14]: Conv_1_Relu_ss[5,14]
   + 21.61040402775194 C1494 <= 21.61040402775194
 Conv_1_Relu_constraint_3[5,15]: Conv_1_Relu_ss[5,15]
   + 11.46170062280965 C1495 <= 11.46170062280965
 Conv_1_Relu_constraint_3[6,0]: Conv_1_Relu_ss[6,0]
   + 56.29245011536772 C1496 <= 56.29245011536772
 Conv_1_Relu_constraint_3[6,1]: Conv_1_Relu_ss[6,1]
   + 67.52969714306539 C1497 <= 67.52969714306539
 Conv_1_Relu_constraint_3[6,2]: Conv_1_Relu_ss[6,2]
   + 88.95482652838119 C1498 <= 88.95482652838119
 Conv_1_Relu_constraint_3[6,3]: Conv_1_Relu_ss[6,3]
   + 33.94596106828702 C1499 <= 33.94596106828702
 Conv_1_Relu_constraint_3[6,4]: Conv_1_Relu_ss[6,4]
   + 106.1251358090966 C1500 <= 106.1251358090966
 Conv_1_Relu_constraint_3[6,5]: Conv_1_Relu_ss[6,5]
   + 10.77560081932031 C1501 <= 10.77560081932031
 Conv_1_Relu_constraint_3[6,6]: Conv_1_Relu_ss[6,6]
   + 29.32218285779035 C1502 <= 29.32218285779035
 Conv_1_Relu_constraint_3[6,7]: Conv_1_Relu_ss[6,7]
   + 24.30472028988515 C1503 <= 24.30472028988515
 Conv_1_Relu_constraint_3[6,8]: Conv_1_Relu_ss[6,8]
   + 25.8812298511837 C1504 <= 25.8812298511837
 Conv_1_Relu_constraint_3[6,9]: Conv_1_Relu_ss[6,9]
   + 76.75665981089216 C1505 <= 76.75665981089216
 Conv_1_Relu_constraint_3[6,10]: Conv_1_Relu_ss[6,10]
   + 54.2754182075725 C1506 <= 54.2754182075725
 Conv_1_Relu_constraint_3[6,11]: Conv_1_Relu_ss[6,11]
   + 97.18723355279275 C1507 <= 97.18723355279275
 Conv_1_Relu_constraint_3[6,12]: Conv_1_Relu_ss[6,12]
   + 47.62093859518428 C1508 <= 47.62093859518428
 Conv_1_Relu_constraint_3[6,13]: Conv_1_Relu_ss[6,13]
   + 21.10965662228852 C1509 <= 21.10965662228852
 Conv_1_Relu_constraint_3[6,14]: Conv_1_Relu_ss[6,14]
   + 21.61040402775194 C1510 <= 21.61040402775194
 Conv_1_Relu_constraint_3[6,15]: Conv_1_Relu_ss[6,15]
   + 11.46170062280965 C1511 <= 11.46170062280965
 Conv_1_Relu_constraint_3[7,0]: Conv_1_Relu_ss[7,0]
   + 56.29245011536772 C1512 <= 56.29245011536772
 Conv_1_Relu_constraint_3[7,1]: Conv_1_Relu_ss[7,1]
   + 67.52969714306539 C1513 <= 67.52969714306539
 Conv_1_Relu_constraint_3[7,2]: Conv_1_Relu_ss[7,2]
   + 88.95482652838119 C1514 <= 88.95482652838119
 Conv_1_Relu_constraint_3[7,3]: Conv_1_Relu_ss[7,3]
   + 33.94596106828702 C1515 <= 33.94596106828702
 Conv_1_Relu_constraint_3[7,4]: Conv_1_Relu_ss[7,4]
   + 106.1251358090966 C1516 <= 106.1251358090966
 Conv_1_Relu_constraint_3[7,5]: Conv_1_Relu_ss[7,5]
   + 10.77560081932031 C1517 <= 10.77560081932031
 Conv_1_Relu_constraint_3[7,6]: Conv_1_Relu_ss[7,6]
   + 29.32218285779035 C1518 <= 29.32218285779035
 Conv_1_Relu_constraint_3[7,7]: Conv_1_Relu_ss[7,7]
   + 24.30472028988515 C1519 <= 24.30472028988515
 Conv_1_Relu_constraint_3[7,8]: Conv_1_Relu_ss[7,8]
   + 25.8812298511837 C1520 <= 25.8812298511837
 Conv_1_Relu_constraint_3[7,9]: Conv_1_Relu_ss[7,9]
   + 76.75665981089216 C1521 <= 76.75665981089216
 Conv_1_Relu_constraint_3[7,10]: Conv_1_Relu_ss[7,10]
   + 54.2754182075725 C1522 <= 54.2754182075725
 Conv_1_Relu_constraint_3[7,11]: Conv_1_Relu_ss[7,11]
   + 97.18723355279275 C1523 <= 97.18723355279275
 Conv_1_Relu_constraint_3[7,12]: Conv_1_Relu_ss[7,12]
   + 47.62093859518428 C1524 <= 47.62093859518428
 Conv_1_Relu_constraint_3[7,13]: Conv_1_Relu_ss[7,13]
   + 21.10965662228852 C1525 <= 21.10965662228852
 Conv_1_Relu_constraint_3[7,14]: Conv_1_Relu_ss[7,14]
   + 21.61040402775194 C1526 <= 21.61040402775194
 Conv_1_Relu_constraint_3[7,15]: Conv_1_Relu_ss[7,15]
   + 11.46170062280965 C1527 <= 11.46170062280965
 Conv_1_Relu_constraint_3[8,0]: Conv_1_Relu_ss[8,0]
   + 56.29245011536772 C1528 <= 56.29245011536772
 Conv_1_Relu_constraint_3[8,1]: Conv_1_Relu_ss[8,1]
   + 67.52969714306539 C1529 <= 67.52969714306539
 Conv_1_Relu_constraint_3[8,2]: Conv_1_Relu_ss[8,2]
   + 88.95482652838119 C1530 <= 88.95482652838119
 Conv_1_Relu_constraint_3[8,3]: Conv_1_Relu_ss[8,3]
   + 33.94596106828702 C1531 <= 33.94596106828702
 Conv_1_Relu_constraint_3[8,4]: Conv_1_Relu_ss[8,4]
   + 106.1251358090966 C1532 <= 106.1251358090966
 Conv_1_Relu_constraint_3[8,5]: Conv_1_Relu_ss[8,5]
   + 10.77560081932031 C1533 <= 10.77560081932031
 Conv_1_Relu_constraint_3[8,6]: Conv_1_Relu_ss[8,6]
   + 29.32218285779035 C1534 <= 29.32218285779035
 Conv_1_Relu_constraint_3[8,7]: Conv_1_Relu_ss[8,7]
   + 24.30472028988515 C1535 <= 24.30472028988515
 Conv_1_Relu_constraint_3[8,8]: Conv_1_Relu_ss[8,8]
   + 25.8812298511837 C1536 <= 25.8812298511837
 Conv_1_Relu_constraint_3[8,9]: Conv_1_Relu_ss[8,9]
   + 76.75665981089216 C1537 <= 76.75665981089216
 Conv_1_Relu_constraint_3[8,10]: Conv_1_Relu_ss[8,10]
   + 54.2754182075725 C1538 <= 54.2754182075725
 Conv_1_Relu_constraint_3[8,11]: Conv_1_Relu_ss[8,11]
   + 97.18723355279275 C1539 <= 97.18723355279275
 Conv_1_Relu_constraint_3[8,12]: Conv_1_Relu_ss[8,12]
   + 47.62093859518428 C1540 <= 47.62093859518428
 Conv_1_Relu_constraint_3[8,13]: Conv_1_Relu_ss[8,13]
   + 21.10965662228852 C1541 <= 21.10965662228852
 Conv_1_Relu_constraint_3[8,14]: Conv_1_Relu_ss[8,14]
   + 21.61040402775194 C1542 <= 21.61040402775194
 Conv_1_Relu_constraint_3[8,15]: Conv_1_Relu_ss[8,15]
   + 11.46170062280965 C1543 <= 11.46170062280965
 Conv_1_Relu_constraint_3[9,0]: Conv_1_Relu_ss[9,0]
   + 56.29245011536772 C1544 <= 56.29245011536772
 Conv_1_Relu_constraint_3[9,1]: Conv_1_Relu_ss[9,1]
   + 67.52969714306539 C1545 <= 67.52969714306539
 Conv_1_Relu_constraint_3[9,2]: Conv_1_Relu_ss[9,2]
   + 88.95482652838119 C1546 <= 88.95482652838119
 Conv_1_Relu_constraint_3[9,3]: Conv_1_Relu_ss[9,3]
   + 33.94596106828702 C1547 <= 33.94596106828702
 Conv_1_Relu_constraint_3[9,4]: Conv_1_Relu_ss[9,4]
   + 106.1251358090966 C1548 <= 106.1251358090966
 Conv_1_Relu_constraint_3[9,5]: Conv_1_Relu_ss[9,5]
   + 10.77560081932031 C1549 <= 10.77560081932031
 Conv_1_Relu_constraint_3[9,6]: Conv_1_Relu_ss[9,6]
   + 29.32218285779035 C1550 <= 29.32218285779035
 Conv_1_Relu_constraint_3[9,7]: Conv_1_Relu_ss[9,7]
   + 24.30472028988515 C1551 <= 24.30472028988515
 Conv_1_Relu_constraint_3[9,8]: Conv_1_Relu_ss[9,8]
   + 25.8812298511837 C1552 <= 25.8812298511837
 Conv_1_Relu_constraint_3[9,9]: Conv_1_Relu_ss[9,9]
   + 76.75665981089216 C1553 <= 76.75665981089216
 Conv_1_Relu_constraint_3[9,10]: Conv_1_Relu_ss[9,10]
   + 54.2754182075725 C1554 <= 54.2754182075725
 Conv_1_Relu_constraint_3[9,11]: Conv_1_Relu_ss[9,11]
   + 97.18723355279275 C1555 <= 97.18723355279275
 Conv_1_Relu_constraint_3[9,12]: Conv_1_Relu_ss[9,12]
   + 47.62093859518428 C1556 <= 47.62093859518428
 Conv_1_Relu_constraint_3[9,13]: Conv_1_Relu_ss[9,13]
   + 21.10965662228852 C1557 <= 21.10965662228852
 Conv_1_Relu_constraint_3[9,14]: Conv_1_Relu_ss[9,14]
   + 21.61040402775194 C1558 <= 21.61040402775194
 Conv_1_Relu_constraint_3[9,15]: Conv_1_Relu_ss[9,15]
   + 11.46170062280965 C1559 <= 11.46170062280965
 Conv_2_output_constraint[0,0]: - 1.283687114715576 Conv_1_Relu_ts[0,0]
   - 0.7330741286277771 Conv_1_Relu_ts[0,1]
   - 2.277222394943237 Conv_1_Relu_ts[0,2]
   + 0.2110073268413544 Conv_1_Relu_ts[0,3]
   - 2.244216442108154 Conv_1_Relu_ts[0,4]
   + 0.1934826672077179 Conv_1_Relu_ts[0,5]
   - 1.220147132873535 Conv_1_Relu_ts[0,6]
   - 0.2545500099658966 Conv_1_Relu_ts[0,7]
   + 1.472832441329956 Conv_1_Relu_ts[0,8]
   - 1.084863781929016 Conv_1_Relu_ts[0,9]
   + 3.159408807754517 Conv_1_Relu_ts[0,10]
   - 1.101905822753906 Conv_1_Relu_ts[0,11]
   + 0.4563636183738708 Conv_1_Relu_ts[0,12]
   - 0.1066205129027367 Conv_1_Relu_ts[0,13]
   + 0.1282547265291214 Conv_1_Relu_ts[0,14]
   - 0.1789750158786774 Conv_1_Relu_ts[0,15]
   + 1.150216221809387 Conv_2_aggregated_features[0,0]
   + 0.7049534916877747 Conv_2_aggregated_features[0,1]
   - 0.6801650524139404 Conv_2_aggregated_features[0,2]
   + 0.0596005544066429 Conv_2_aggregated_features[0,3]
   + 1.205108404159546 Conv_2_aggregated_features[0,4]
   - 0.0892623364925385 Conv_2_aggregated_features[0,5]
   + 1.500906825065613 Conv_2_aggregated_features[0,6]
   - 0.4810487031936646 Conv_2_aggregated_features[0,7]
   + 0.3540807068347931 Conv_2_aggregated_features[0,8]
   - 0.3831521570682526 Conv_2_aggregated_features[0,9]
   - 0.5447250604629517 Conv_2_aggregated_features[0,10]
   + 2.773675441741943 Conv_2_aggregated_features[0,11]
   + 0.4864743649959564 Conv_2_aggregated_features[0,12]
   + 0.2153479158878326 Conv_2_aggregated_features[0,13]
   + 1.067057013511658 Conv_2_aggregated_features[0,14]
   - 0.2962089478969574 Conv_2_aggregated_features[0,15] + Conv_2_t[0,0]
   = -0.1796741187572479
 Conv_2_output_constraint[0,1]: - 0.8503658175468445 Conv_1_Relu_ts[0,0]
   + 0.1598657667636871 Conv_1_Relu_ts[0,1]
   + 0.740400493144989 Conv_1_Relu_ts[0,2]
   - 0.0143840508535504 Conv_1_Relu_ts[0,3]
   + 0.418368011713028 Conv_1_Relu_ts[0,4]
   - 0.0816935896873474 Conv_1_Relu_ts[0,5]
   - 0.6345715522766113 Conv_1_Relu_ts[0,6]
   - 0.1658433526754379 Conv_1_Relu_ts[0,7]
   + 0.2588719427585602 Conv_1_Relu_ts[0,8]
   - 0.3725545704364777 Conv_1_Relu_ts[0,9]
   - 0.1093620359897614 Conv_1_Relu_ts[0,10]
   - 0.229639545083046 Conv_1_Relu_ts[0,11]
   + 2.353754758834839 Conv_1_Relu_ts[0,12]
   + 0.275741845369339 Conv_1_Relu_ts[0,13]
   + 0.2292672693729401 Conv_1_Relu_ts[0,14]
   - 0.085102416574955 Conv_1_Relu_ts[0,15]
   + 1.31701123714447 Conv_2_aggregated_features[0,0]
   + 0.5385994911193848 Conv_2_aggregated_features[0,1]
   - 0.8459239602088928 Conv_2_aggregated_features[0,2]
   + 0.1865081340074539 Conv_2_aggregated_features[0,3]
   + 0.8385969400405884 Conv_2_aggregated_features[0,4]
   - 0.1273856163024902 Conv_2_aggregated_features[0,5]
   + 2.53032374382019 Conv_2_aggregated_features[0,6]
   - 0.3543893694877625 Conv_2_aggregated_features[0,7]
   + 0.1923532038927078 Conv_2_aggregated_features[0,8]
   + 1.131453990936279 Conv_2_aggregated_features[0,9]
   - 0.3651714324951172 Conv_2_aggregated_features[0,10]
   + 1.542536735534668 Conv_2_aggregated_features[0,11]
   - 0.4610330760478973 Conv_2_aggregated_features[0,12]
   - 0.00123762537259609 Conv_2_aggregated_features[0,13]
   + 0.375784307718277 Conv_2_aggregated_features[0,14]
   + 0.0822948440909386 Conv_2_aggregated_features[0,15] + Conv_2_t[0,1]
   = -0.280759334564209
 Conv_2_output_constraint[0,2]: 1.746561527252197 Conv_1_Relu_ts[0,0]
   + 0.814217746257782 Conv_1_Relu_ts[0,1]
   - 1.465980052947998 Conv_1_Relu_ts[0,2]
   - 0.2278070449829102 Conv_1_Relu_ts[0,3]
   + 1.037705659866333 Conv_1_Relu_ts[0,4]
   - 0.2538289427757263 Conv_1_Relu_ts[0,5]
   + 0.6437492966651917 Conv_1_Relu_ts[0,6]
   - 0.4347417056560516 Conv_1_Relu_ts[0,7]
   + 0.2060042917728424 Conv_1_Relu_ts[0,8]
   + 0.8382987976074219 Conv_1_Relu_ts[0,9]
   - 0.6877544522285461 Conv_1_Relu_ts[0,10]
   + 1.350347280502319 Conv_1_Relu_ts[0,11]
   + 2.458039999008179 Conv_1_Relu_ts[0,12]
   - 0.1695821583271027 Conv_1_Relu_ts[0,13]
   + 0.2783661186695099 Conv_1_Relu_ts[0,14]
   - 0.1666328907012939 Conv_1_Relu_ts[0,15]
   - 0.4272331893444061 Conv_2_aggregated_features[0,0]
   + 0.8438286781311035 Conv_2_aggregated_features[0,1]
   - 1.847808957099915 Conv_2_aggregated_features[0,2]
   - 0.1186513304710388 Conv_2_aggregated_features[0,3]
   + 0.6336349844932556 Conv_2_aggregated_features[0,4]
   + 0.0602188408374786 Conv_2_aggregated_features[0,5]
   - 0.2530380189418793 Conv_2_aggregated_features[0,6]
   - 0.5760374069213867 Conv_2_aggregated_features[0,7]
   - 0.8055176734924316 Conv_2_aggregated_features[0,8]
   + 0.4366684854030609 Conv_2_aggregated_features[0,9]
   - 1.177173972129822 Conv_2_aggregated_features[0,10]
   - 0.4784670472145081 Conv_2_aggregated_features[0,11]
   + 2.325191736221313 Conv_2_aggregated_features[0,12]
   - 0.2128628194332123 Conv_2_aggregated_features[0,13]
   - 0.1453570574522018 Conv_2_aggregated_features[0,14]
   - 0.118608683347702 Conv_2_aggregated_features[0,15] + Conv_2_t[0,2]
   = 1.507851004600525
 Conv_2_output_constraint[0,3]: - 0.527165412902832 Conv_1_Relu_ts[0,0]
   - 1.389649629592896 Conv_1_Relu_ts[0,1]
   - 0.6640920639038086 Conv_1_Relu_ts[0,2]
   - 0.0634000822901726 Conv_1_Relu_ts[0,3]
   - 0.4163890480995178 Conv_1_Relu_ts[0,4]
   + 0.0613848567008972 Conv_1_Relu_ts[0,5]
   + 1.790452718734741 Conv_1_Relu_ts[0,6]
   - 0.1128331422805786 Conv_1_Relu_ts[0,7]
   + 2.12717342376709 Conv_1_Relu_ts[0,8]
   - 0.1405593752861023 Conv_1_Relu_ts[0,9]
   - 0.1852237284183502 Conv_1_Relu_ts[0,10]
   - 0.2454802542924881 Conv_1_Relu_ts[0,11]
   - 1.97015655040741 Conv_1_Relu_ts[0,12]
   + 0.3291732370853424 Conv_1_Relu_ts[0,13]
   - 0.5780640244483948 Conv_1_Relu_ts[0,14]
   - 0.1067387536168098 Conv_1_Relu_ts[0,15]
   - 0.1432812511920929 Conv_2_aggregated_features[0,0]
   - 0.4248973727226257 Conv_2_aggregated_features[0,1]
   + 1.125378131866455 Conv_2_aggregated_features[0,2]
   + 0.1260246187448502 Conv_2_aggregated_features[0,3]
   - 0.5538251996040344 Conv_2_aggregated_features[0,4]
   + 0.1312114000320435 Conv_2_aggregated_features[0,5]
   - 0.9671168327331543 Conv_2_aggregated_features[0,6]
   - 0.0926898792386055 Conv_2_aggregated_features[0,7]
   - 0.1030022576451302 Conv_2_aggregated_features[0,8]
   - 1.69473397731781 Conv_2_aggregated_features[0,9]
   + 2.313785552978516 Conv_2_aggregated_features[0,10]
   - 0.3551321625709534 Conv_2_aggregated_features[0,11]
   - 0.441056102514267 Conv_2_aggregated_features[0,12]
   - 0.0765185728669167 Conv_2_aggregated_features[0,13]
   - 0.5983367562294006 Conv_2_aggregated_features[0,14]
   + 0.023831807076931 Conv_2_aggregated_features[0,15] + Conv_2_t[0,3]
   = -0.072962760925293
 Conv_2_output_constraint[0,4]: 0.5219888687133789 Conv_1_Relu_ts[0,0]
   - 1.025602698326111 Conv_1_Relu_ts[0,1]
   - 0.4347750842571259 Conv_1_Relu_ts[0,2]
   + 0.2773195803165436 Conv_1_Relu_ts[0,3]
   - 2.5947265625 Conv_1_Relu_ts[0,4]
   - 0.2330431044101715 Conv_1_Relu_ts[0,5]
   + 0.4634701013565063 Conv_1_Relu_ts[0,6]
   - 0.2591606974601746 Conv_1_Relu_ts[0,7]
   + 0.715503990650177 Conv_1_Relu_ts[0,8]
   - 0.1986619830131531 Conv_1_Relu_ts[0,9]
   - 0.9099530577659607 Conv_1_Relu_ts[0,10]
   - 0.2335229516029358 Conv_1_Relu_ts[0,11]
   - 0.2992779910564423 Conv_1_Relu_ts[0,12]
   + 0.000100307 Conv_1_Relu_ts[0,13]
   - 1.00616979598999 Conv_1_Relu_ts[0,14]
   + 0.2120778858661652 Conv_1_Relu_ts[0,15]
   - 0.0600817315280437 Conv_2_aggregated_features[0,0]
   - 0.099200963973999 Conv_2_aggregated_features[0,1]
   - 0.4625514149665833 Conv_2_aggregated_features[0,2]
   - 0.1394201517105103 Conv_2_aggregated_features[0,3]
   + 1.153588056564331 Conv_2_aggregated_features[0,4]
   - 0.0127062797546387 Conv_2_aggregated_features[0,5]
   + 1.360898375511169 Conv_2_aggregated_features[0,6]
   - 0.1619014590978622 Conv_2_aggregated_features[0,7]
   + 0.0715495049953461 Conv_2_aggregated_features[0,8]
   + 0.1082091927528381 Conv_2_aggregated_features[0,9]
   + 1.219732761383057 Conv_2_aggregated_features[0,10]
   + 0.0949473306536674 Conv_2_aggregated_features[0,11]
   - 0.7561001181602478 Conv_2_aggregated_features[0,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[0,13]
   + 0.3085470497608185 Conv_2_aggregated_features[0,14]
   - 0.1963151693344116 Conv_2_aggregated_features[0,15] + Conv_2_t[0,4]
   = -0.681087851524353
 Conv_2_output_constraint[0,5]: 0.4891856014728546 Conv_1_Relu_ts[0,0]
   - 1.764690637588501 Conv_1_Relu_ts[0,1]
   + 1.572006106376648 Conv_1_Relu_ts[0,2]
   - 0.2603900134563446 Conv_1_Relu_ts[0,3]
   - 2.532854557037354 Conv_1_Relu_ts[0,4]
   - 0.0787733495235443 Conv_1_Relu_ts[0,5]
   + 0.3314015865325928 Conv_1_Relu_ts[0,6]
   + 0.3532952666282654 Conv_1_Relu_ts[0,7]
   + 1.495306849479675 Conv_1_Relu_ts[0,8]
   - 0.3582539260387421 Conv_1_Relu_ts[0,9]
   + 1.239243626594543 Conv_1_Relu_ts[0,10]
   + 0.0796256512403488 Conv_1_Relu_ts[0,11]
   + 1.406325340270996 Conv_1_Relu_ts[0,12]
   + 0.2874834537506104 Conv_1_Relu_ts[0,13]
   - 0.4502094686031342 Conv_1_Relu_ts[0,14]
   - 0.1813227832317352 Conv_1_Relu_ts[0,15]
   - 1.002788901329041 Conv_2_aggregated_features[0,0]
   - 1.268097281455994 Conv_2_aggregated_features[0,1]
   - 1.182242274284363 Conv_2_aggregated_features[0,2]
   - 0.3146973550319672 Conv_2_aggregated_features[0,3]
   + 1.659744143486023 Conv_2_aggregated_features[0,4]
   - 0.0238673985004425 Conv_2_aggregated_features[0,5]
   - 0.0993486419320107 Conv_2_aggregated_features[0,6]
   + 0.1021591797471046 Conv_2_aggregated_features[0,7]
   - 0.9812391996383667 Conv_2_aggregated_features[0,8]
   + 0.6361307501792908 Conv_2_aggregated_features[0,9]
   - 0.2150242179632187 Conv_2_aggregated_features[0,10]
   - 2.391498804092407 Conv_2_aggregated_features[0,11]
   + 0.9408105611801147 Conv_2_aggregated_features[0,12]
   - 0.0835467427968979 Conv_2_aggregated_features[0,13]
   - 0.7100914716720581 Conv_2_aggregated_features[0,14]
   + 0.081059992313385 Conv_2_aggregated_features[0,15] + Conv_2_t[0,5]
   = -1.552430987358093
 Conv_2_output_constraint[0,6]: 0.3276324272155762 Conv_1_Relu_ts[0,0]
   - 1.526854515075684 Conv_1_Relu_ts[0,1]
   - 0.2327080518007278 Conv_1_Relu_ts[0,2]
   + 0.2350454330444336 Conv_1_Relu_ts[0,3]
   - 2.336396932601929 Conv_1_Relu_ts[0,4]
   + 0.0498765707015991 Conv_1_Relu_ts[0,5]
   + 0.1361224502325058 Conv_1_Relu_ts[0,6]
   - 0.7505534887313843 Conv_1_Relu_ts[0,7]
   - 0.1081981733441353 Conv_1_Relu_ts[0,8]
   + 0.36195108294487 Conv_1_Relu_ts[0,9]
   + 0.266346663236618 Conv_1_Relu_ts[0,10]
   - 0.2529528141021729 Conv_1_Relu_ts[0,11]
   + 0.3400895595550537 Conv_1_Relu_ts[0,12]
   - 0.1297148019075394 Conv_1_Relu_ts[0,13]
   + 0.2151342928409576 Conv_1_Relu_ts[0,14]
   - 0.0570834763348103 Conv_1_Relu_ts[0,15]
   - 0.2712872326374054 Conv_2_aggregated_features[0,0]
   - 0.6653857827186584 Conv_2_aggregated_features[0,1]
   + 0.0704844743013382 Conv_2_aggregated_features[0,2]
   - 0.00456566177308559 Conv_2_aggregated_features[0,3]
   + 0.0726276636123657 Conv_2_aggregated_features[0,4]
   - 0.1622210741043091 Conv_2_aggregated_features[0,5]
   - 0.6965416669845581 Conv_2_aggregated_features[0,6]
   - 0.7083758115768433 Conv_2_aggregated_features[0,7]
   - 0.6968990564346313 Conv_2_aggregated_features[0,8]
   + 0.575663149356842 Conv_2_aggregated_features[0,9]
   - 0.3281366527080536 Conv_2_aggregated_features[0,10]
   - 1.37471342086792 Conv_2_aggregated_features[0,11]
   + 1.073476433753967 Conv_2_aggregated_features[0,12]
   + 0.0794747099280357 Conv_2_aggregated_features[0,13]
   + 0.5443784594535828 Conv_2_aggregated_features[0,14]
   - 0.0632283836603165 Conv_2_aggregated_features[0,15] + Conv_2_t[0,6]
   = 0.4907519221305847
 Conv_2_output_constraint[0,7]: - 0.0696223676204681 Conv_1_Relu_ts[0,0]
   - 0.1037913709878922 Conv_1_Relu_ts[0,1]
   + 0.5115458965301514 Conv_1_Relu_ts[0,2]
   - 0.1024313941597939 Conv_1_Relu_ts[0,3]
   + 0.2925211489200592 Conv_1_Relu_ts[0,4]
   + 0.1706541776657104 Conv_1_Relu_ts[0,5]
   + 0.1667102426290512 Conv_1_Relu_ts[0,6]
   + 0.2208153456449509 Conv_1_Relu_ts[0,7]
   + 0.0217245072126389 Conv_1_Relu_ts[0,8]
   + 0.2856383323669434 Conv_1_Relu_ts[0,9]
   + 0.1799796521663666 Conv_1_Relu_ts[0,10]
   + 0.2511521875858307 Conv_1_Relu_ts[0,11]
   + 0.1590938121080399 Conv_1_Relu_ts[0,12]
   - 0.1085780262947083 Conv_1_Relu_ts[0,13]
   - 0.1072387397289276 Conv_1_Relu_ts[0,14]
   + 0.0257974341511726 Conv_1_Relu_ts[0,15]
   - 0.00193481589667499 Conv_2_aggregated_features[0,0]
   - 0.0832197070121765 Conv_2_aggregated_features[0,1]
   + 0.2962732911109924 Conv_2_aggregated_features[0,2]
   + 0.1245143488049507 Conv_2_aggregated_features[0,3]
   + 0.0374857746064663 Conv_2_aggregated_features[0,4]
   - 0.2035515308380127 Conv_2_aggregated_features[0,5]
   + 0.0183010119944811 Conv_2_aggregated_features[0,6]
   - 0.0808482989668846 Conv_2_aggregated_features[0,7]
   + 0.0491516962647438 Conv_2_aggregated_features[0,8]
   - 0.1665307730436325 Conv_2_aggregated_features[0,9]
   + 0.180585503578186 Conv_2_aggregated_features[0,10]
   + 0.0900064706802368 Conv_2_aggregated_features[0,11]
   + 0.1489291191101074 Conv_2_aggregated_features[0,12]
   - 0.1824352592229843 Conv_2_aggregated_features[0,13]
   - 0.0335522592067719 Conv_2_aggregated_features[0,14]
   + 0.1226971745491028 Conv_2_aggregated_features[0,15] + Conv_2_t[0,7]
   = -0.0930573865771294
 Conv_2_output_constraint[1,0]: - 1.283687114715576 Conv_1_Relu_ts[1,0]
   - 0.7330741286277771 Conv_1_Relu_ts[1,1]
   - 2.277222394943237 Conv_1_Relu_ts[1,2]
   + 0.2110073268413544 Conv_1_Relu_ts[1,3]
   - 2.244216442108154 Conv_1_Relu_ts[1,4]
   + 0.1934826672077179 Conv_1_Relu_ts[1,5]
   - 1.220147132873535 Conv_1_Relu_ts[1,6]
   - 0.2545500099658966 Conv_1_Relu_ts[1,7]
   + 1.472832441329956 Conv_1_Relu_ts[1,8]
   - 1.084863781929016 Conv_1_Relu_ts[1,9]
   + 3.159408807754517 Conv_1_Relu_ts[1,10]
   - 1.101905822753906 Conv_1_Relu_ts[1,11]
   + 0.4563636183738708 Conv_1_Relu_ts[1,12]
   - 0.1066205129027367 Conv_1_Relu_ts[1,13]
   + 0.1282547265291214 Conv_1_Relu_ts[1,14]
   - 0.1789750158786774 Conv_1_Relu_ts[1,15]
   + 1.150216221809387 Conv_2_aggregated_features[1,0]
   + 0.7049534916877747 Conv_2_aggregated_features[1,1]
   - 0.6801650524139404 Conv_2_aggregated_features[1,2]
   + 0.0596005544066429 Conv_2_aggregated_features[1,3]
   + 1.205108404159546 Conv_2_aggregated_features[1,4]
   - 0.0892623364925385 Conv_2_aggregated_features[1,5]
   + 1.500906825065613 Conv_2_aggregated_features[1,6]
   - 0.4810487031936646 Conv_2_aggregated_features[1,7]
   + 0.3540807068347931 Conv_2_aggregated_features[1,8]
   - 0.3831521570682526 Conv_2_aggregated_features[1,9]
   - 0.5447250604629517 Conv_2_aggregated_features[1,10]
   + 2.773675441741943 Conv_2_aggregated_features[1,11]
   + 0.4864743649959564 Conv_2_aggregated_features[1,12]
   + 0.2153479158878326 Conv_2_aggregated_features[1,13]
   + 1.067057013511658 Conv_2_aggregated_features[1,14]
   - 0.2962089478969574 Conv_2_aggregated_features[1,15] + Conv_2_t[1,0]
   = -0.1796741187572479
 Conv_2_output_constraint[1,1]: - 0.8503658175468445 Conv_1_Relu_ts[1,0]
   + 0.1598657667636871 Conv_1_Relu_ts[1,1]
   + 0.740400493144989 Conv_1_Relu_ts[1,2]
   - 0.0143840508535504 Conv_1_Relu_ts[1,3]
   + 0.418368011713028 Conv_1_Relu_ts[1,4]
   - 0.0816935896873474 Conv_1_Relu_ts[1,5]
   - 0.6345715522766113 Conv_1_Relu_ts[1,6]
   - 0.1658433526754379 Conv_1_Relu_ts[1,7]
   + 0.2588719427585602 Conv_1_Relu_ts[1,8]
   - 0.3725545704364777 Conv_1_Relu_ts[1,9]
   - 0.1093620359897614 Conv_1_Relu_ts[1,10]
   - 0.229639545083046 Conv_1_Relu_ts[1,11]
   + 2.353754758834839 Conv_1_Relu_ts[1,12]
   + 0.275741845369339 Conv_1_Relu_ts[1,13]
   + 0.2292672693729401 Conv_1_Relu_ts[1,14]
   - 0.085102416574955 Conv_1_Relu_ts[1,15]
   + 1.31701123714447 Conv_2_aggregated_features[1,0]
   + 0.5385994911193848 Conv_2_aggregated_features[1,1]
   - 0.8459239602088928 Conv_2_aggregated_features[1,2]
   + 0.1865081340074539 Conv_2_aggregated_features[1,3]
   + 0.8385969400405884 Conv_2_aggregated_features[1,4]
   - 0.1273856163024902 Conv_2_aggregated_features[1,5]
   + 2.53032374382019 Conv_2_aggregated_features[1,6]
   - 0.3543893694877625 Conv_2_aggregated_features[1,7]
   + 0.1923532038927078 Conv_2_aggregated_features[1,8]
   + 1.131453990936279 Conv_2_aggregated_features[1,9]
   - 0.3651714324951172 Conv_2_aggregated_features[1,10]
   + 1.542536735534668 Conv_2_aggregated_features[1,11]
   - 0.4610330760478973 Conv_2_aggregated_features[1,12]
   - 0.00123762537259609 Conv_2_aggregated_features[1,13]
   + 0.375784307718277 Conv_2_aggregated_features[1,14]
   + 0.0822948440909386 Conv_2_aggregated_features[1,15] + Conv_2_t[1,1]
   = -0.280759334564209
 Conv_2_output_constraint[1,2]: 1.746561527252197 Conv_1_Relu_ts[1,0]
   + 0.814217746257782 Conv_1_Relu_ts[1,1]
   - 1.465980052947998 Conv_1_Relu_ts[1,2]
   - 0.2278070449829102 Conv_1_Relu_ts[1,3]
   + 1.037705659866333 Conv_1_Relu_ts[1,4]
   - 0.2538289427757263 Conv_1_Relu_ts[1,5]
   + 0.6437492966651917 Conv_1_Relu_ts[1,6]
   - 0.4347417056560516 Conv_1_Relu_ts[1,7]
   + 0.2060042917728424 Conv_1_Relu_ts[1,8]
   + 0.8382987976074219 Conv_1_Relu_ts[1,9]
   - 0.6877544522285461 Conv_1_Relu_ts[1,10]
   + 1.350347280502319 Conv_1_Relu_ts[1,11]
   + 2.458039999008179 Conv_1_Relu_ts[1,12]
   - 0.1695821583271027 Conv_1_Relu_ts[1,13]
   + 0.2783661186695099 Conv_1_Relu_ts[1,14]
   - 0.1666328907012939 Conv_1_Relu_ts[1,15]
   - 0.4272331893444061 Conv_2_aggregated_features[1,0]
   + 0.8438286781311035 Conv_2_aggregated_features[1,1]
   - 1.847808957099915 Conv_2_aggregated_features[1,2]
   - 0.1186513304710388 Conv_2_aggregated_features[1,3]
   + 0.6336349844932556 Conv_2_aggregated_features[1,4]
   + 0.0602188408374786 Conv_2_aggregated_features[1,5]
   - 0.2530380189418793 Conv_2_aggregated_features[1,6]
   - 0.5760374069213867 Conv_2_aggregated_features[1,7]
   - 0.8055176734924316 Conv_2_aggregated_features[1,8]
   + 0.4366684854030609 Conv_2_aggregated_features[1,9]
   - 1.177173972129822 Conv_2_aggregated_features[1,10]
   - 0.4784670472145081 Conv_2_aggregated_features[1,11]
   + 2.325191736221313 Conv_2_aggregated_features[1,12]
   - 0.2128628194332123 Conv_2_aggregated_features[1,13]
   - 0.1453570574522018 Conv_2_aggregated_features[1,14]
   - 0.118608683347702 Conv_2_aggregated_features[1,15] + Conv_2_t[1,2]
   = 1.507851004600525
 Conv_2_output_constraint[1,3]: - 0.527165412902832 Conv_1_Relu_ts[1,0]
   - 1.389649629592896 Conv_1_Relu_ts[1,1]
   - 0.6640920639038086 Conv_1_Relu_ts[1,2]
   - 0.0634000822901726 Conv_1_Relu_ts[1,3]
   - 0.4163890480995178 Conv_1_Relu_ts[1,4]
   + 0.0613848567008972 Conv_1_Relu_ts[1,5]
   + 1.790452718734741 Conv_1_Relu_ts[1,6]
   - 0.1128331422805786 Conv_1_Relu_ts[1,7]
   + 2.12717342376709 Conv_1_Relu_ts[1,8]
   - 0.1405593752861023 Conv_1_Relu_ts[1,9]
   - 0.1852237284183502 Conv_1_Relu_ts[1,10]
   - 0.2454802542924881 Conv_1_Relu_ts[1,11]
   - 1.97015655040741 Conv_1_Relu_ts[1,12]
   + 0.3291732370853424 Conv_1_Relu_ts[1,13]
   - 0.5780640244483948 Conv_1_Relu_ts[1,14]
   - 0.1067387536168098 Conv_1_Relu_ts[1,15]
   - 0.1432812511920929 Conv_2_aggregated_features[1,0]
   - 0.4248973727226257 Conv_2_aggregated_features[1,1]
   + 1.125378131866455 Conv_2_aggregated_features[1,2]
   + 0.1260246187448502 Conv_2_aggregated_features[1,3]
   - 0.5538251996040344 Conv_2_aggregated_features[1,4]
   + 0.1312114000320435 Conv_2_aggregated_features[1,5]
   - 0.9671168327331543 Conv_2_aggregated_features[1,6]
   - 0.0926898792386055 Conv_2_aggregated_features[1,7]
   - 0.1030022576451302 Conv_2_aggregated_features[1,8]
   - 1.69473397731781 Conv_2_aggregated_features[1,9]
   + 2.313785552978516 Conv_2_aggregated_features[1,10]
   - 0.3551321625709534 Conv_2_aggregated_features[1,11]
   - 0.441056102514267 Conv_2_aggregated_features[1,12]
   - 0.0765185728669167 Conv_2_aggregated_features[1,13]
   - 0.5983367562294006 Conv_2_aggregated_features[1,14]
   + 0.023831807076931 Conv_2_aggregated_features[1,15] + Conv_2_t[1,3]
   = -0.072962760925293
 Conv_2_output_constraint[1,4]: 0.5219888687133789 Conv_1_Relu_ts[1,0]
   - 1.025602698326111 Conv_1_Relu_ts[1,1]
   - 0.4347750842571259 Conv_1_Relu_ts[1,2]
   + 0.2773195803165436 Conv_1_Relu_ts[1,3]
   - 2.5947265625 Conv_1_Relu_ts[1,4]
   - 0.2330431044101715 Conv_1_Relu_ts[1,5]
   + 0.4634701013565063 Conv_1_Relu_ts[1,6]
   - 0.2591606974601746 Conv_1_Relu_ts[1,7]
   + 0.715503990650177 Conv_1_Relu_ts[1,8]
   - 0.1986619830131531 Conv_1_Relu_ts[1,9]
   - 0.9099530577659607 Conv_1_Relu_ts[1,10]
   - 0.2335229516029358 Conv_1_Relu_ts[1,11]
   - 0.2992779910564423 Conv_1_Relu_ts[1,12]
   + 0.000100307 Conv_1_Relu_ts[1,13]
   - 1.00616979598999 Conv_1_Relu_ts[1,14]
   + 0.2120778858661652 Conv_1_Relu_ts[1,15]
   - 0.0600817315280437 Conv_2_aggregated_features[1,0]
   - 0.099200963973999 Conv_2_aggregated_features[1,1]
   - 0.4625514149665833 Conv_2_aggregated_features[1,2]
   - 0.1394201517105103 Conv_2_aggregated_features[1,3]
   + 1.153588056564331 Conv_2_aggregated_features[1,4]
   - 0.0127062797546387 Conv_2_aggregated_features[1,5]
   + 1.360898375511169 Conv_2_aggregated_features[1,6]
   - 0.1619014590978622 Conv_2_aggregated_features[1,7]
   + 0.0715495049953461 Conv_2_aggregated_features[1,8]
   + 0.1082091927528381 Conv_2_aggregated_features[1,9]
   + 1.219732761383057 Conv_2_aggregated_features[1,10]
   + 0.0949473306536674 Conv_2_aggregated_features[1,11]
   - 0.7561001181602478 Conv_2_aggregated_features[1,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[1,13]
   + 0.3085470497608185 Conv_2_aggregated_features[1,14]
   - 0.1963151693344116 Conv_2_aggregated_features[1,15] + Conv_2_t[1,4]
   = -0.681087851524353
 Conv_2_output_constraint[1,5]: 0.4891856014728546 Conv_1_Relu_ts[1,0]
   - 1.764690637588501 Conv_1_Relu_ts[1,1]
   + 1.572006106376648 Conv_1_Relu_ts[1,2]
   - 0.2603900134563446 Conv_1_Relu_ts[1,3]
   - 2.532854557037354 Conv_1_Relu_ts[1,4]
   - 0.0787733495235443 Conv_1_Relu_ts[1,5]
   + 0.3314015865325928 Conv_1_Relu_ts[1,6]
   + 0.3532952666282654 Conv_1_Relu_ts[1,7]
   + 1.495306849479675 Conv_1_Relu_ts[1,8]
   - 0.3582539260387421 Conv_1_Relu_ts[1,9]
   + 1.239243626594543 Conv_1_Relu_ts[1,10]
   + 0.0796256512403488 Conv_1_Relu_ts[1,11]
   + 1.406325340270996 Conv_1_Relu_ts[1,12]
   + 0.2874834537506104 Conv_1_Relu_ts[1,13]
   - 0.4502094686031342 Conv_1_Relu_ts[1,14]
   - 0.1813227832317352 Conv_1_Relu_ts[1,15]
   - 1.002788901329041 Conv_2_aggregated_features[1,0]
   - 1.268097281455994 Conv_2_aggregated_features[1,1]
   - 1.182242274284363 Conv_2_aggregated_features[1,2]
   - 0.3146973550319672 Conv_2_aggregated_features[1,3]
   + 1.659744143486023 Conv_2_aggregated_features[1,4]
   - 0.0238673985004425 Conv_2_aggregated_features[1,5]
   - 0.0993486419320107 Conv_2_aggregated_features[1,6]
   + 0.1021591797471046 Conv_2_aggregated_features[1,7]
   - 0.9812391996383667 Conv_2_aggregated_features[1,8]
   + 0.6361307501792908 Conv_2_aggregated_features[1,9]
   - 0.2150242179632187 Conv_2_aggregated_features[1,10]
   - 2.391498804092407 Conv_2_aggregated_features[1,11]
   + 0.9408105611801147 Conv_2_aggregated_features[1,12]
   - 0.0835467427968979 Conv_2_aggregated_features[1,13]
   - 0.7100914716720581 Conv_2_aggregated_features[1,14]
   + 0.081059992313385 Conv_2_aggregated_features[1,15] + Conv_2_t[1,5]
   = -1.552430987358093
 Conv_2_output_constraint[1,6]: 0.3276324272155762 Conv_1_Relu_ts[1,0]
   - 1.526854515075684 Conv_1_Relu_ts[1,1]
   - 0.2327080518007278 Conv_1_Relu_ts[1,2]
   + 0.2350454330444336 Conv_1_Relu_ts[1,3]
   - 2.336396932601929 Conv_1_Relu_ts[1,4]
   + 0.0498765707015991 Conv_1_Relu_ts[1,5]
   + 0.1361224502325058 Conv_1_Relu_ts[1,6]
   - 0.7505534887313843 Conv_1_Relu_ts[1,7]
   - 0.1081981733441353 Conv_1_Relu_ts[1,8]
   + 0.36195108294487 Conv_1_Relu_ts[1,9]
   + 0.266346663236618 Conv_1_Relu_ts[1,10]
   - 0.2529528141021729 Conv_1_Relu_ts[1,11]
   + 0.3400895595550537 Conv_1_Relu_ts[1,12]
   - 0.1297148019075394 Conv_1_Relu_ts[1,13]
   + 0.2151342928409576 Conv_1_Relu_ts[1,14]
   - 0.0570834763348103 Conv_1_Relu_ts[1,15]
   - 0.2712872326374054 Conv_2_aggregated_features[1,0]
   - 0.6653857827186584 Conv_2_aggregated_features[1,1]
   + 0.0704844743013382 Conv_2_aggregated_features[1,2]
   - 0.00456566177308559 Conv_2_aggregated_features[1,3]
   + 0.0726276636123657 Conv_2_aggregated_features[1,4]
   - 0.1622210741043091 Conv_2_aggregated_features[1,5]
   - 0.6965416669845581 Conv_2_aggregated_features[1,6]
   - 0.7083758115768433 Conv_2_aggregated_features[1,7]
   - 0.6968990564346313 Conv_2_aggregated_features[1,8]
   + 0.575663149356842 Conv_2_aggregated_features[1,9]
   - 0.3281366527080536 Conv_2_aggregated_features[1,10]
   - 1.37471342086792 Conv_2_aggregated_features[1,11]
   + 1.073476433753967 Conv_2_aggregated_features[1,12]
   + 0.0794747099280357 Conv_2_aggregated_features[1,13]
   + 0.5443784594535828 Conv_2_aggregated_features[1,14]
   - 0.0632283836603165 Conv_2_aggregated_features[1,15] + Conv_2_t[1,6]
   = 0.4907519221305847
 Conv_2_output_constraint[1,7]: - 0.0696223676204681 Conv_1_Relu_ts[1,0]
   - 0.1037913709878922 Conv_1_Relu_ts[1,1]
   + 0.5115458965301514 Conv_1_Relu_ts[1,2]
   - 0.1024313941597939 Conv_1_Relu_ts[1,3]
   + 0.2925211489200592 Conv_1_Relu_ts[1,4]
   + 0.1706541776657104 Conv_1_Relu_ts[1,5]
   + 0.1667102426290512 Conv_1_Relu_ts[1,6]
   + 0.2208153456449509 Conv_1_Relu_ts[1,7]
   + 0.0217245072126389 Conv_1_Relu_ts[1,8]
   + 0.2856383323669434 Conv_1_Relu_ts[1,9]
   + 0.1799796521663666 Conv_1_Relu_ts[1,10]
   + 0.2511521875858307 Conv_1_Relu_ts[1,11]
   + 0.1590938121080399 Conv_1_Relu_ts[1,12]
   - 0.1085780262947083 Conv_1_Relu_ts[1,13]
   - 0.1072387397289276 Conv_1_Relu_ts[1,14]
   + 0.0257974341511726 Conv_1_Relu_ts[1,15]
   - 0.00193481589667499 Conv_2_aggregated_features[1,0]
   - 0.0832197070121765 Conv_2_aggregated_features[1,1]
   + 0.2962732911109924 Conv_2_aggregated_features[1,2]
   + 0.1245143488049507 Conv_2_aggregated_features[1,3]
   + 0.0374857746064663 Conv_2_aggregated_features[1,4]
   - 0.2035515308380127 Conv_2_aggregated_features[1,5]
   + 0.0183010119944811 Conv_2_aggregated_features[1,6]
   - 0.0808482989668846 Conv_2_aggregated_features[1,7]
   + 0.0491516962647438 Conv_2_aggregated_features[1,8]
   - 0.1665307730436325 Conv_2_aggregated_features[1,9]
   + 0.180585503578186 Conv_2_aggregated_features[1,10]
   + 0.0900064706802368 Conv_2_aggregated_features[1,11]
   + 0.1489291191101074 Conv_2_aggregated_features[1,12]
   - 0.1824352592229843 Conv_2_aggregated_features[1,13]
   - 0.0335522592067719 Conv_2_aggregated_features[1,14]
   + 0.1226971745491028 Conv_2_aggregated_features[1,15] + Conv_2_t[1,7]
   = -0.0930573865771294
 Conv_2_output_constraint[2,0]: - 1.283687114715576 Conv_1_Relu_ts[2,0]
   - 0.7330741286277771 Conv_1_Relu_ts[2,1]
   - 2.277222394943237 Conv_1_Relu_ts[2,2]
   + 0.2110073268413544 Conv_1_Relu_ts[2,3]
   - 2.244216442108154 Conv_1_Relu_ts[2,4]
   + 0.1934826672077179 Conv_1_Relu_ts[2,5]
   - 1.220147132873535 Conv_1_Relu_ts[2,6]
   - 0.2545500099658966 Conv_1_Relu_ts[2,7]
   + 1.472832441329956 Conv_1_Relu_ts[2,8]
   - 1.084863781929016 Conv_1_Relu_ts[2,9]
   + 3.159408807754517 Conv_1_Relu_ts[2,10]
   - 1.101905822753906 Conv_1_Relu_ts[2,11]
   + 0.4563636183738708 Conv_1_Relu_ts[2,12]
   - 0.1066205129027367 Conv_1_Relu_ts[2,13]
   + 0.1282547265291214 Conv_1_Relu_ts[2,14]
   - 0.1789750158786774 Conv_1_Relu_ts[2,15]
   + 1.150216221809387 Conv_2_aggregated_features[2,0]
   + 0.7049534916877747 Conv_2_aggregated_features[2,1]
   - 0.6801650524139404 Conv_2_aggregated_features[2,2]
   + 0.0596005544066429 Conv_2_aggregated_features[2,3]
   + 1.205108404159546 Conv_2_aggregated_features[2,4]
   - 0.0892623364925385 Conv_2_aggregated_features[2,5]
   + 1.500906825065613 Conv_2_aggregated_features[2,6]
   - 0.4810487031936646 Conv_2_aggregated_features[2,7]
   + 0.3540807068347931 Conv_2_aggregated_features[2,8]
   - 0.3831521570682526 Conv_2_aggregated_features[2,9]
   - 0.5447250604629517 Conv_2_aggregated_features[2,10]
   + 2.773675441741943 Conv_2_aggregated_features[2,11]
   + 0.4864743649959564 Conv_2_aggregated_features[2,12]
   + 0.2153479158878326 Conv_2_aggregated_features[2,13]
   + 1.067057013511658 Conv_2_aggregated_features[2,14]
   - 0.2962089478969574 Conv_2_aggregated_features[2,15] + Conv_2_t[2,0]
   = -0.1796741187572479
 Conv_2_output_constraint[2,1]: - 0.8503658175468445 Conv_1_Relu_ts[2,0]
   + 0.1598657667636871 Conv_1_Relu_ts[2,1]
   + 0.740400493144989 Conv_1_Relu_ts[2,2]
   - 0.0143840508535504 Conv_1_Relu_ts[2,3]
   + 0.418368011713028 Conv_1_Relu_ts[2,4]
   - 0.0816935896873474 Conv_1_Relu_ts[2,5]
   - 0.6345715522766113 Conv_1_Relu_ts[2,6]
   - 0.1658433526754379 Conv_1_Relu_ts[2,7]
   + 0.2588719427585602 Conv_1_Relu_ts[2,8]
   - 0.3725545704364777 Conv_1_Relu_ts[2,9]
   - 0.1093620359897614 Conv_1_Relu_ts[2,10]
   - 0.229639545083046 Conv_1_Relu_ts[2,11]
   + 2.353754758834839 Conv_1_Relu_ts[2,12]
   + 0.275741845369339 Conv_1_Relu_ts[2,13]
   + 0.2292672693729401 Conv_1_Relu_ts[2,14]
   - 0.085102416574955 Conv_1_Relu_ts[2,15]
   + 1.31701123714447 Conv_2_aggregated_features[2,0]
   + 0.5385994911193848 Conv_2_aggregated_features[2,1]
   - 0.8459239602088928 Conv_2_aggregated_features[2,2]
   + 0.1865081340074539 Conv_2_aggregated_features[2,3]
   + 0.8385969400405884 Conv_2_aggregated_features[2,4]
   - 0.1273856163024902 Conv_2_aggregated_features[2,5]
   + 2.53032374382019 Conv_2_aggregated_features[2,6]
   - 0.3543893694877625 Conv_2_aggregated_features[2,7]
   + 0.1923532038927078 Conv_2_aggregated_features[2,8]
   + 1.131453990936279 Conv_2_aggregated_features[2,9]
   - 0.3651714324951172 Conv_2_aggregated_features[2,10]
   + 1.542536735534668 Conv_2_aggregated_features[2,11]
   - 0.4610330760478973 Conv_2_aggregated_features[2,12]
   - 0.00123762537259609 Conv_2_aggregated_features[2,13]
   + 0.375784307718277 Conv_2_aggregated_features[2,14]
   + 0.0822948440909386 Conv_2_aggregated_features[2,15] + Conv_2_t[2,1]
   = -0.280759334564209
 Conv_2_output_constraint[2,2]: 1.746561527252197 Conv_1_Relu_ts[2,0]
   + 0.814217746257782 Conv_1_Relu_ts[2,1]
   - 1.465980052947998 Conv_1_Relu_ts[2,2]
   - 0.2278070449829102 Conv_1_Relu_ts[2,3]
   + 1.037705659866333 Conv_1_Relu_ts[2,4]
   - 0.2538289427757263 Conv_1_Relu_ts[2,5]
   + 0.6437492966651917 Conv_1_Relu_ts[2,6]
   - 0.4347417056560516 Conv_1_Relu_ts[2,7]
   + 0.2060042917728424 Conv_1_Relu_ts[2,8]
   + 0.8382987976074219 Conv_1_Relu_ts[2,9]
   - 0.6877544522285461 Conv_1_Relu_ts[2,10]
   + 1.350347280502319 Conv_1_Relu_ts[2,11]
   + 2.458039999008179 Conv_1_Relu_ts[2,12]
   - 0.1695821583271027 Conv_1_Relu_ts[2,13]
   + 0.2783661186695099 Conv_1_Relu_ts[2,14]
   - 0.1666328907012939 Conv_1_Relu_ts[2,15]
   - 0.4272331893444061 Conv_2_aggregated_features[2,0]
   + 0.8438286781311035 Conv_2_aggregated_features[2,1]
   - 1.847808957099915 Conv_2_aggregated_features[2,2]
   - 0.1186513304710388 Conv_2_aggregated_features[2,3]
   + 0.6336349844932556 Conv_2_aggregated_features[2,4]
   + 0.0602188408374786 Conv_2_aggregated_features[2,5]
   - 0.2530380189418793 Conv_2_aggregated_features[2,6]
   - 0.5760374069213867 Conv_2_aggregated_features[2,7]
   - 0.8055176734924316 Conv_2_aggregated_features[2,8]
   + 0.4366684854030609 Conv_2_aggregated_features[2,9]
   - 1.177173972129822 Conv_2_aggregated_features[2,10]
   - 0.4784670472145081 Conv_2_aggregated_features[2,11]
   + 2.325191736221313 Conv_2_aggregated_features[2,12]
   - 0.2128628194332123 Conv_2_aggregated_features[2,13]
   - 0.1453570574522018 Conv_2_aggregated_features[2,14]
   - 0.118608683347702 Conv_2_aggregated_features[2,15] + Conv_2_t[2,2]
   = 1.507851004600525
 Conv_2_output_constraint[2,3]: - 0.527165412902832 Conv_1_Relu_ts[2,0]
   - 1.389649629592896 Conv_1_Relu_ts[2,1]
   - 0.6640920639038086 Conv_1_Relu_ts[2,2]
   - 0.0634000822901726 Conv_1_Relu_ts[2,3]
   - 0.4163890480995178 Conv_1_Relu_ts[2,4]
   + 0.0613848567008972 Conv_1_Relu_ts[2,5]
   + 1.790452718734741 Conv_1_Relu_ts[2,6]
   - 0.1128331422805786 Conv_1_Relu_ts[2,7]
   + 2.12717342376709 Conv_1_Relu_ts[2,8]
   - 0.1405593752861023 Conv_1_Relu_ts[2,9]
   - 0.1852237284183502 Conv_1_Relu_ts[2,10]
   - 0.2454802542924881 Conv_1_Relu_ts[2,11]
   - 1.97015655040741 Conv_1_Relu_ts[2,12]
   + 0.3291732370853424 Conv_1_Relu_ts[2,13]
   - 0.5780640244483948 Conv_1_Relu_ts[2,14]
   - 0.1067387536168098 Conv_1_Relu_ts[2,15]
   - 0.1432812511920929 Conv_2_aggregated_features[2,0]
   - 0.4248973727226257 Conv_2_aggregated_features[2,1]
   + 1.125378131866455 Conv_2_aggregated_features[2,2]
   + 0.1260246187448502 Conv_2_aggregated_features[2,3]
   - 0.5538251996040344 Conv_2_aggregated_features[2,4]
   + 0.1312114000320435 Conv_2_aggregated_features[2,5]
   - 0.9671168327331543 Conv_2_aggregated_features[2,6]
   - 0.0926898792386055 Conv_2_aggregated_features[2,7]
   - 0.1030022576451302 Conv_2_aggregated_features[2,8]
   - 1.69473397731781 Conv_2_aggregated_features[2,9]
   + 2.313785552978516 Conv_2_aggregated_features[2,10]
   - 0.3551321625709534 Conv_2_aggregated_features[2,11]
   - 0.441056102514267 Conv_2_aggregated_features[2,12]
   - 0.0765185728669167 Conv_2_aggregated_features[2,13]
   - 0.5983367562294006 Conv_2_aggregated_features[2,14]
   + 0.023831807076931 Conv_2_aggregated_features[2,15] + Conv_2_t[2,3]
   = -0.072962760925293
 Conv_2_output_constraint[2,4]: 0.5219888687133789 Conv_1_Relu_ts[2,0]
   - 1.025602698326111 Conv_1_Relu_ts[2,1]
   - 0.4347750842571259 Conv_1_Relu_ts[2,2]
   + 0.2773195803165436 Conv_1_Relu_ts[2,3]
   - 2.5947265625 Conv_1_Relu_ts[2,4]
   - 0.2330431044101715 Conv_1_Relu_ts[2,5]
   + 0.4634701013565063 Conv_1_Relu_ts[2,6]
   - 0.2591606974601746 Conv_1_Relu_ts[2,7]
   + 0.715503990650177 Conv_1_Relu_ts[2,8]
   - 0.1986619830131531 Conv_1_Relu_ts[2,9]
   - 0.9099530577659607 Conv_1_Relu_ts[2,10]
   - 0.2335229516029358 Conv_1_Relu_ts[2,11]
   - 0.2992779910564423 Conv_1_Relu_ts[2,12]
   + 0.000100307 Conv_1_Relu_ts[2,13]
   - 1.00616979598999 Conv_1_Relu_ts[2,14]
   + 0.2120778858661652 Conv_1_Relu_ts[2,15]
   - 0.0600817315280437 Conv_2_aggregated_features[2,0]
   - 0.099200963973999 Conv_2_aggregated_features[2,1]
   - 0.4625514149665833 Conv_2_aggregated_features[2,2]
   - 0.1394201517105103 Conv_2_aggregated_features[2,3]
   + 1.153588056564331 Conv_2_aggregated_features[2,4]
   - 0.0127062797546387 Conv_2_aggregated_features[2,5]
   + 1.360898375511169 Conv_2_aggregated_features[2,6]
   - 0.1619014590978622 Conv_2_aggregated_features[2,7]
   + 0.0715495049953461 Conv_2_aggregated_features[2,8]
   + 0.1082091927528381 Conv_2_aggregated_features[2,9]
   + 1.219732761383057 Conv_2_aggregated_features[2,10]
   + 0.0949473306536674 Conv_2_aggregated_features[2,11]
   - 0.7561001181602478 Conv_2_aggregated_features[2,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[2,13]
   + 0.3085470497608185 Conv_2_aggregated_features[2,14]
   - 0.1963151693344116 Conv_2_aggregated_features[2,15] + Conv_2_t[2,4]
   = -0.681087851524353
 Conv_2_output_constraint[2,5]: 0.4891856014728546 Conv_1_Relu_ts[2,0]
   - 1.764690637588501 Conv_1_Relu_ts[2,1]
   + 1.572006106376648 Conv_1_Relu_ts[2,2]
   - 0.2603900134563446 Conv_1_Relu_ts[2,3]
   - 2.532854557037354 Conv_1_Relu_ts[2,4]
   - 0.0787733495235443 Conv_1_Relu_ts[2,5]
   + 0.3314015865325928 Conv_1_Relu_ts[2,6]
   + 0.3532952666282654 Conv_1_Relu_ts[2,7]
   + 1.495306849479675 Conv_1_Relu_ts[2,8]
   - 0.3582539260387421 Conv_1_Relu_ts[2,9]
   + 1.239243626594543 Conv_1_Relu_ts[2,10]
   + 0.0796256512403488 Conv_1_Relu_ts[2,11]
   + 1.406325340270996 Conv_1_Relu_ts[2,12]
   + 0.2874834537506104 Conv_1_Relu_ts[2,13]
   - 0.4502094686031342 Conv_1_Relu_ts[2,14]
   - 0.1813227832317352 Conv_1_Relu_ts[2,15]
   - 1.002788901329041 Conv_2_aggregated_features[2,0]
   - 1.268097281455994 Conv_2_aggregated_features[2,1]
   - 1.182242274284363 Conv_2_aggregated_features[2,2]
   - 0.3146973550319672 Conv_2_aggregated_features[2,3]
   + 1.659744143486023 Conv_2_aggregated_features[2,4]
   - 0.0238673985004425 Conv_2_aggregated_features[2,5]
   - 0.0993486419320107 Conv_2_aggregated_features[2,6]
   + 0.1021591797471046 Conv_2_aggregated_features[2,7]
   - 0.9812391996383667 Conv_2_aggregated_features[2,8]
   + 0.6361307501792908 Conv_2_aggregated_features[2,9]
   - 0.2150242179632187 Conv_2_aggregated_features[2,10]
   - 2.391498804092407 Conv_2_aggregated_features[2,11]
   + 0.9408105611801147 Conv_2_aggregated_features[2,12]
   - 0.0835467427968979 Conv_2_aggregated_features[2,13]
   - 0.7100914716720581 Conv_2_aggregated_features[2,14]
   + 0.081059992313385 Conv_2_aggregated_features[2,15] + Conv_2_t[2,5]
   = -1.552430987358093
 Conv_2_output_constraint[2,6]: 0.3276324272155762 Conv_1_Relu_ts[2,0]
   - 1.526854515075684 Conv_1_Relu_ts[2,1]
   - 0.2327080518007278 Conv_1_Relu_ts[2,2]
   + 0.2350454330444336 Conv_1_Relu_ts[2,3]
   - 2.336396932601929 Conv_1_Relu_ts[2,4]
   + 0.0498765707015991 Conv_1_Relu_ts[2,5]
   + 0.1361224502325058 Conv_1_Relu_ts[2,6]
   - 0.7505534887313843 Conv_1_Relu_ts[2,7]
   - 0.1081981733441353 Conv_1_Relu_ts[2,8]
   + 0.36195108294487 Conv_1_Relu_ts[2,9]
   + 0.266346663236618 Conv_1_Relu_ts[2,10]
   - 0.2529528141021729 Conv_1_Relu_ts[2,11]
   + 0.3400895595550537 Conv_1_Relu_ts[2,12]
   - 0.1297148019075394 Conv_1_Relu_ts[2,13]
   + 0.2151342928409576 Conv_1_Relu_ts[2,14]
   - 0.0570834763348103 Conv_1_Relu_ts[2,15]
   - 0.2712872326374054 Conv_2_aggregated_features[2,0]
   - 0.6653857827186584 Conv_2_aggregated_features[2,1]
   + 0.0704844743013382 Conv_2_aggregated_features[2,2]
   - 0.00456566177308559 Conv_2_aggregated_features[2,3]
   + 0.0726276636123657 Conv_2_aggregated_features[2,4]
   - 0.1622210741043091 Conv_2_aggregated_features[2,5]
   - 0.6965416669845581 Conv_2_aggregated_features[2,6]
   - 0.7083758115768433 Conv_2_aggregated_features[2,7]
   - 0.6968990564346313 Conv_2_aggregated_features[2,8]
   + 0.575663149356842 Conv_2_aggregated_features[2,9]
   - 0.3281366527080536 Conv_2_aggregated_features[2,10]
   - 1.37471342086792 Conv_2_aggregated_features[2,11]
   + 1.073476433753967 Conv_2_aggregated_features[2,12]
   + 0.0794747099280357 Conv_2_aggregated_features[2,13]
   + 0.5443784594535828 Conv_2_aggregated_features[2,14]
   - 0.0632283836603165 Conv_2_aggregated_features[2,15] + Conv_2_t[2,6]
   = 0.4907519221305847
 Conv_2_output_constraint[2,7]: - 0.0696223676204681 Conv_1_Relu_ts[2,0]
   - 0.1037913709878922 Conv_1_Relu_ts[2,1]
   + 0.5115458965301514 Conv_1_Relu_ts[2,2]
   - 0.1024313941597939 Conv_1_Relu_ts[2,3]
   + 0.2925211489200592 Conv_1_Relu_ts[2,4]
   + 0.1706541776657104 Conv_1_Relu_ts[2,5]
   + 0.1667102426290512 Conv_1_Relu_ts[2,6]
   + 0.2208153456449509 Conv_1_Relu_ts[2,7]
   + 0.0217245072126389 Conv_1_Relu_ts[2,8]
   + 0.2856383323669434 Conv_1_Relu_ts[2,9]
   + 0.1799796521663666 Conv_1_Relu_ts[2,10]
   + 0.2511521875858307 Conv_1_Relu_ts[2,11]
   + 0.1590938121080399 Conv_1_Relu_ts[2,12]
   - 0.1085780262947083 Conv_1_Relu_ts[2,13]
   - 0.1072387397289276 Conv_1_Relu_ts[2,14]
   + 0.0257974341511726 Conv_1_Relu_ts[2,15]
   - 0.00193481589667499 Conv_2_aggregated_features[2,0]
   - 0.0832197070121765 Conv_2_aggregated_features[2,1]
   + 0.2962732911109924 Conv_2_aggregated_features[2,2]
   + 0.1245143488049507 Conv_2_aggregated_features[2,3]
   + 0.0374857746064663 Conv_2_aggregated_features[2,4]
   - 0.2035515308380127 Conv_2_aggregated_features[2,5]
   + 0.0183010119944811 Conv_2_aggregated_features[2,6]
   - 0.0808482989668846 Conv_2_aggregated_features[2,7]
   + 0.0491516962647438 Conv_2_aggregated_features[2,8]
   - 0.1665307730436325 Conv_2_aggregated_features[2,9]
   + 0.180585503578186 Conv_2_aggregated_features[2,10]
   + 0.0900064706802368 Conv_2_aggregated_features[2,11]
   + 0.1489291191101074 Conv_2_aggregated_features[2,12]
   - 0.1824352592229843 Conv_2_aggregated_features[2,13]
   - 0.0335522592067719 Conv_2_aggregated_features[2,14]
   + 0.1226971745491028 Conv_2_aggregated_features[2,15] + Conv_2_t[2,7]
   = -0.0930573865771294
 Conv_2_output_constraint[3,0]: - 1.283687114715576 Conv_1_Relu_ts[3,0]
   - 0.7330741286277771 Conv_1_Relu_ts[3,1]
   - 2.277222394943237 Conv_1_Relu_ts[3,2]
   + 0.2110073268413544 Conv_1_Relu_ts[3,3]
   - 2.244216442108154 Conv_1_Relu_ts[3,4]
   + 0.1934826672077179 Conv_1_Relu_ts[3,5]
   - 1.220147132873535 Conv_1_Relu_ts[3,6]
   - 0.2545500099658966 Conv_1_Relu_ts[3,7]
   + 1.472832441329956 Conv_1_Relu_ts[3,8]
   - 1.084863781929016 Conv_1_Relu_ts[3,9]
   + 3.159408807754517 Conv_1_Relu_ts[3,10]
   - 1.101905822753906 Conv_1_Relu_ts[3,11]
   + 0.4563636183738708 Conv_1_Relu_ts[3,12]
   - 0.1066205129027367 Conv_1_Relu_ts[3,13]
   + 0.1282547265291214 Conv_1_Relu_ts[3,14]
   - 0.1789750158786774 Conv_1_Relu_ts[3,15]
   + 1.150216221809387 Conv_2_aggregated_features[3,0]
   + 0.7049534916877747 Conv_2_aggregated_features[3,1]
   - 0.6801650524139404 Conv_2_aggregated_features[3,2]
   + 0.0596005544066429 Conv_2_aggregated_features[3,3]
   + 1.205108404159546 Conv_2_aggregated_features[3,4]
   - 0.0892623364925385 Conv_2_aggregated_features[3,5]
   + 1.500906825065613 Conv_2_aggregated_features[3,6]
   - 0.4810487031936646 Conv_2_aggregated_features[3,7]
   + 0.3540807068347931 Conv_2_aggregated_features[3,8]
   - 0.3831521570682526 Conv_2_aggregated_features[3,9]
   - 0.5447250604629517 Conv_2_aggregated_features[3,10]
   + 2.773675441741943 Conv_2_aggregated_features[3,11]
   + 0.4864743649959564 Conv_2_aggregated_features[3,12]
   + 0.2153479158878326 Conv_2_aggregated_features[3,13]
   + 1.067057013511658 Conv_2_aggregated_features[3,14]
   - 0.2962089478969574 Conv_2_aggregated_features[3,15] + Conv_2_t[3,0]
   = -0.1796741187572479
 Conv_2_output_constraint[3,1]: - 0.8503658175468445 Conv_1_Relu_ts[3,0]
   + 0.1598657667636871 Conv_1_Relu_ts[3,1]
   + 0.740400493144989 Conv_1_Relu_ts[3,2]
   - 0.0143840508535504 Conv_1_Relu_ts[3,3]
   + 0.418368011713028 Conv_1_Relu_ts[3,4]
   - 0.0816935896873474 Conv_1_Relu_ts[3,5]
   - 0.6345715522766113 Conv_1_Relu_ts[3,6]
   - 0.1658433526754379 Conv_1_Relu_ts[3,7]
   + 0.2588719427585602 Conv_1_Relu_ts[3,8]
   - 0.3725545704364777 Conv_1_Relu_ts[3,9]
   - 0.1093620359897614 Conv_1_Relu_ts[3,10]
   - 0.229639545083046 Conv_1_Relu_ts[3,11]
   + 2.353754758834839 Conv_1_Relu_ts[3,12]
   + 0.275741845369339 Conv_1_Relu_ts[3,13]
   + 0.2292672693729401 Conv_1_Relu_ts[3,14]
   - 0.085102416574955 Conv_1_Relu_ts[3,15]
   + 1.31701123714447 Conv_2_aggregated_features[3,0]
   + 0.5385994911193848 Conv_2_aggregated_features[3,1]
   - 0.8459239602088928 Conv_2_aggregated_features[3,2]
   + 0.1865081340074539 Conv_2_aggregated_features[3,3]
   + 0.8385969400405884 Conv_2_aggregated_features[3,4]
   - 0.1273856163024902 Conv_2_aggregated_features[3,5]
   + 2.53032374382019 Conv_2_aggregated_features[3,6]
   - 0.3543893694877625 Conv_2_aggregated_features[3,7]
   + 0.1923532038927078 Conv_2_aggregated_features[3,8]
   + 1.131453990936279 Conv_2_aggregated_features[3,9]
   - 0.3651714324951172 Conv_2_aggregated_features[3,10]
   + 1.542536735534668 Conv_2_aggregated_features[3,11]
   - 0.4610330760478973 Conv_2_aggregated_features[3,12]
   - 0.00123762537259609 Conv_2_aggregated_features[3,13]
   + 0.375784307718277 Conv_2_aggregated_features[3,14]
   + 0.0822948440909386 Conv_2_aggregated_features[3,15] + Conv_2_t[3,1]
   = -0.280759334564209
 Conv_2_output_constraint[3,2]: 1.746561527252197 Conv_1_Relu_ts[3,0]
   + 0.814217746257782 Conv_1_Relu_ts[3,1]
   - 1.465980052947998 Conv_1_Relu_ts[3,2]
   - 0.2278070449829102 Conv_1_Relu_ts[3,3]
   + 1.037705659866333 Conv_1_Relu_ts[3,4]
   - 0.2538289427757263 Conv_1_Relu_ts[3,5]
   + 0.6437492966651917 Conv_1_Relu_ts[3,6]
   - 0.4347417056560516 Conv_1_Relu_ts[3,7]
   + 0.2060042917728424 Conv_1_Relu_ts[3,8]
   + 0.8382987976074219 Conv_1_Relu_ts[3,9]
   - 0.6877544522285461 Conv_1_Relu_ts[3,10]
   + 1.350347280502319 Conv_1_Relu_ts[3,11]
   + 2.458039999008179 Conv_1_Relu_ts[3,12]
   - 0.1695821583271027 Conv_1_Relu_ts[3,13]
   + 0.2783661186695099 Conv_1_Relu_ts[3,14]
   - 0.1666328907012939 Conv_1_Relu_ts[3,15]
   - 0.4272331893444061 Conv_2_aggregated_features[3,0]
   + 0.8438286781311035 Conv_2_aggregated_features[3,1]
   - 1.847808957099915 Conv_2_aggregated_features[3,2]
   - 0.1186513304710388 Conv_2_aggregated_features[3,3]
   + 0.6336349844932556 Conv_2_aggregated_features[3,4]
   + 0.0602188408374786 Conv_2_aggregated_features[3,5]
   - 0.2530380189418793 Conv_2_aggregated_features[3,6]
   - 0.5760374069213867 Conv_2_aggregated_features[3,7]
   - 0.8055176734924316 Conv_2_aggregated_features[3,8]
   + 0.4366684854030609 Conv_2_aggregated_features[3,9]
   - 1.177173972129822 Conv_2_aggregated_features[3,10]
   - 0.4784670472145081 Conv_2_aggregated_features[3,11]
   + 2.325191736221313 Conv_2_aggregated_features[3,12]
   - 0.2128628194332123 Conv_2_aggregated_features[3,13]
   - 0.1453570574522018 Conv_2_aggregated_features[3,14]
   - 0.118608683347702 Conv_2_aggregated_features[3,15] + Conv_2_t[3,2]
   = 1.507851004600525
 Conv_2_output_constraint[3,3]: - 0.527165412902832 Conv_1_Relu_ts[3,0]
   - 1.389649629592896 Conv_1_Relu_ts[3,1]
   - 0.6640920639038086 Conv_1_Relu_ts[3,2]
   - 0.0634000822901726 Conv_1_Relu_ts[3,3]
   - 0.4163890480995178 Conv_1_Relu_ts[3,4]
   + 0.0613848567008972 Conv_1_Relu_ts[3,5]
   + 1.790452718734741 Conv_1_Relu_ts[3,6]
   - 0.1128331422805786 Conv_1_Relu_ts[3,7]
   + 2.12717342376709 Conv_1_Relu_ts[3,8]
   - 0.1405593752861023 Conv_1_Relu_ts[3,9]
   - 0.1852237284183502 Conv_1_Relu_ts[3,10]
   - 0.2454802542924881 Conv_1_Relu_ts[3,11]
   - 1.97015655040741 Conv_1_Relu_ts[3,12]
   + 0.3291732370853424 Conv_1_Relu_ts[3,13]
   - 0.5780640244483948 Conv_1_Relu_ts[3,14]
   - 0.1067387536168098 Conv_1_Relu_ts[3,15]
   - 0.1432812511920929 Conv_2_aggregated_features[3,0]
   - 0.4248973727226257 Conv_2_aggregated_features[3,1]
   + 1.125378131866455 Conv_2_aggregated_features[3,2]
   + 0.1260246187448502 Conv_2_aggregated_features[3,3]
   - 0.5538251996040344 Conv_2_aggregated_features[3,4]
   + 0.1312114000320435 Conv_2_aggregated_features[3,5]
   - 0.9671168327331543 Conv_2_aggregated_features[3,6]
   - 0.0926898792386055 Conv_2_aggregated_features[3,7]
   - 0.1030022576451302 Conv_2_aggregated_features[3,8]
   - 1.69473397731781 Conv_2_aggregated_features[3,9]
   + 2.313785552978516 Conv_2_aggregated_features[3,10]
   - 0.3551321625709534 Conv_2_aggregated_features[3,11]
   - 0.441056102514267 Conv_2_aggregated_features[3,12]
   - 0.0765185728669167 Conv_2_aggregated_features[3,13]
   - 0.5983367562294006 Conv_2_aggregated_features[3,14]
   + 0.023831807076931 Conv_2_aggregated_features[3,15] + Conv_2_t[3,3]
   = -0.072962760925293
 Conv_2_output_constraint[3,4]: 0.5219888687133789 Conv_1_Relu_ts[3,0]
   - 1.025602698326111 Conv_1_Relu_ts[3,1]
   - 0.4347750842571259 Conv_1_Relu_ts[3,2]
   + 0.2773195803165436 Conv_1_Relu_ts[3,3]
   - 2.5947265625 Conv_1_Relu_ts[3,4]
   - 0.2330431044101715 Conv_1_Relu_ts[3,5]
   + 0.4634701013565063 Conv_1_Relu_ts[3,6]
   - 0.2591606974601746 Conv_1_Relu_ts[3,7]
   + 0.715503990650177 Conv_1_Relu_ts[3,8]
   - 0.1986619830131531 Conv_1_Relu_ts[3,9]
   - 0.9099530577659607 Conv_1_Relu_ts[3,10]
   - 0.2335229516029358 Conv_1_Relu_ts[3,11]
   - 0.2992779910564423 Conv_1_Relu_ts[3,12]
   + 0.000100307 Conv_1_Relu_ts[3,13]
   - 1.00616979598999 Conv_1_Relu_ts[3,14]
   + 0.2120778858661652 Conv_1_Relu_ts[3,15]
   - 0.0600817315280437 Conv_2_aggregated_features[3,0]
   - 0.099200963973999 Conv_2_aggregated_features[3,1]
   - 0.4625514149665833 Conv_2_aggregated_features[3,2]
   - 0.1394201517105103 Conv_2_aggregated_features[3,3]
   + 1.153588056564331 Conv_2_aggregated_features[3,4]
   - 0.0127062797546387 Conv_2_aggregated_features[3,5]
   + 1.360898375511169 Conv_2_aggregated_features[3,6]
   - 0.1619014590978622 Conv_2_aggregated_features[3,7]
   + 0.0715495049953461 Conv_2_aggregated_features[3,8]
   + 0.1082091927528381 Conv_2_aggregated_features[3,9]
   + 1.219732761383057 Conv_2_aggregated_features[3,10]
   + 0.0949473306536674 Conv_2_aggregated_features[3,11]
   - 0.7561001181602478 Conv_2_aggregated_features[3,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[3,13]
   + 0.3085470497608185 Conv_2_aggregated_features[3,14]
   - 0.1963151693344116 Conv_2_aggregated_features[3,15] + Conv_2_t[3,4]
   = -0.681087851524353
 Conv_2_output_constraint[3,5]: 0.4891856014728546 Conv_1_Relu_ts[3,0]
   - 1.764690637588501 Conv_1_Relu_ts[3,1]
   + 1.572006106376648 Conv_1_Relu_ts[3,2]
   - 0.2603900134563446 Conv_1_Relu_ts[3,3]
   - 2.532854557037354 Conv_1_Relu_ts[3,4]
   - 0.0787733495235443 Conv_1_Relu_ts[3,5]
   + 0.3314015865325928 Conv_1_Relu_ts[3,6]
   + 0.3532952666282654 Conv_1_Relu_ts[3,7]
   + 1.495306849479675 Conv_1_Relu_ts[3,8]
   - 0.3582539260387421 Conv_1_Relu_ts[3,9]
   + 1.239243626594543 Conv_1_Relu_ts[3,10]
   + 0.0796256512403488 Conv_1_Relu_ts[3,11]
   + 1.406325340270996 Conv_1_Relu_ts[3,12]
   + 0.2874834537506104 Conv_1_Relu_ts[3,13]
   - 0.4502094686031342 Conv_1_Relu_ts[3,14]
   - 0.1813227832317352 Conv_1_Relu_ts[3,15]
   - 1.002788901329041 Conv_2_aggregated_features[3,0]
   - 1.268097281455994 Conv_2_aggregated_features[3,1]
   - 1.182242274284363 Conv_2_aggregated_features[3,2]
   - 0.3146973550319672 Conv_2_aggregated_features[3,3]
   + 1.659744143486023 Conv_2_aggregated_features[3,4]
   - 0.0238673985004425 Conv_2_aggregated_features[3,5]
   - 0.0993486419320107 Conv_2_aggregated_features[3,6]
   + 0.1021591797471046 Conv_2_aggregated_features[3,7]
   - 0.9812391996383667 Conv_2_aggregated_features[3,8]
   + 0.6361307501792908 Conv_2_aggregated_features[3,9]
   - 0.2150242179632187 Conv_2_aggregated_features[3,10]
   - 2.391498804092407 Conv_2_aggregated_features[3,11]
   + 0.9408105611801147 Conv_2_aggregated_features[3,12]
   - 0.0835467427968979 Conv_2_aggregated_features[3,13]
   - 0.7100914716720581 Conv_2_aggregated_features[3,14]
   + 0.081059992313385 Conv_2_aggregated_features[3,15] + Conv_2_t[3,5]
   = -1.552430987358093
 Conv_2_output_constraint[3,6]: 0.3276324272155762 Conv_1_Relu_ts[3,0]
   - 1.526854515075684 Conv_1_Relu_ts[3,1]
   - 0.2327080518007278 Conv_1_Relu_ts[3,2]
   + 0.2350454330444336 Conv_1_Relu_ts[3,3]
   - 2.336396932601929 Conv_1_Relu_ts[3,4]
   + 0.0498765707015991 Conv_1_Relu_ts[3,5]
   + 0.1361224502325058 Conv_1_Relu_ts[3,6]
   - 0.7505534887313843 Conv_1_Relu_ts[3,7]
   - 0.1081981733441353 Conv_1_Relu_ts[3,8]
   + 0.36195108294487 Conv_1_Relu_ts[3,9]
   + 0.266346663236618 Conv_1_Relu_ts[3,10]
   - 0.2529528141021729 Conv_1_Relu_ts[3,11]
   + 0.3400895595550537 Conv_1_Relu_ts[3,12]
   - 0.1297148019075394 Conv_1_Relu_ts[3,13]
   + 0.2151342928409576 Conv_1_Relu_ts[3,14]
   - 0.0570834763348103 Conv_1_Relu_ts[3,15]
   - 0.2712872326374054 Conv_2_aggregated_features[3,0]
   - 0.6653857827186584 Conv_2_aggregated_features[3,1]
   + 0.0704844743013382 Conv_2_aggregated_features[3,2]
   - 0.00456566177308559 Conv_2_aggregated_features[3,3]
   + 0.0726276636123657 Conv_2_aggregated_features[3,4]
   - 0.1622210741043091 Conv_2_aggregated_features[3,5]
   - 0.6965416669845581 Conv_2_aggregated_features[3,6]
   - 0.7083758115768433 Conv_2_aggregated_features[3,7]
   - 0.6968990564346313 Conv_2_aggregated_features[3,8]
   + 0.575663149356842 Conv_2_aggregated_features[3,9]
   - 0.3281366527080536 Conv_2_aggregated_features[3,10]
   - 1.37471342086792 Conv_2_aggregated_features[3,11]
   + 1.073476433753967 Conv_2_aggregated_features[3,12]
   + 0.0794747099280357 Conv_2_aggregated_features[3,13]
   + 0.5443784594535828 Conv_2_aggregated_features[3,14]
   - 0.0632283836603165 Conv_2_aggregated_features[3,15] + Conv_2_t[3,6]
   = 0.4907519221305847
 Conv_2_output_constraint[3,7]: - 0.0696223676204681 Conv_1_Relu_ts[3,0]
   - 0.1037913709878922 Conv_1_Relu_ts[3,1]
   + 0.5115458965301514 Conv_1_Relu_ts[3,2]
   - 0.1024313941597939 Conv_1_Relu_ts[3,3]
   + 0.2925211489200592 Conv_1_Relu_ts[3,4]
   + 0.1706541776657104 Conv_1_Relu_ts[3,5]
   + 0.1667102426290512 Conv_1_Relu_ts[3,6]
   + 0.2208153456449509 Conv_1_Relu_ts[3,7]
   + 0.0217245072126389 Conv_1_Relu_ts[3,8]
   + 0.2856383323669434 Conv_1_Relu_ts[3,9]
   + 0.1799796521663666 Conv_1_Relu_ts[3,10]
   + 0.2511521875858307 Conv_1_Relu_ts[3,11]
   + 0.1590938121080399 Conv_1_Relu_ts[3,12]
   - 0.1085780262947083 Conv_1_Relu_ts[3,13]
   - 0.1072387397289276 Conv_1_Relu_ts[3,14]
   + 0.0257974341511726 Conv_1_Relu_ts[3,15]
   - 0.00193481589667499 Conv_2_aggregated_features[3,0]
   - 0.0832197070121765 Conv_2_aggregated_features[3,1]
   + 0.2962732911109924 Conv_2_aggregated_features[3,2]
   + 0.1245143488049507 Conv_2_aggregated_features[3,3]
   + 0.0374857746064663 Conv_2_aggregated_features[3,4]
   - 0.2035515308380127 Conv_2_aggregated_features[3,5]
   + 0.0183010119944811 Conv_2_aggregated_features[3,6]
   - 0.0808482989668846 Conv_2_aggregated_features[3,7]
   + 0.0491516962647438 Conv_2_aggregated_features[3,8]
   - 0.1665307730436325 Conv_2_aggregated_features[3,9]
   + 0.180585503578186 Conv_2_aggregated_features[3,10]
   + 0.0900064706802368 Conv_2_aggregated_features[3,11]
   + 0.1489291191101074 Conv_2_aggregated_features[3,12]
   - 0.1824352592229843 Conv_2_aggregated_features[3,13]
   - 0.0335522592067719 Conv_2_aggregated_features[3,14]
   + 0.1226971745491028 Conv_2_aggregated_features[3,15] + Conv_2_t[3,7]
   = -0.0930573865771294
 Conv_2_output_constraint[4,0]: - 1.283687114715576 Conv_1_Relu_ts[4,0]
   - 0.7330741286277771 Conv_1_Relu_ts[4,1]
   - 2.277222394943237 Conv_1_Relu_ts[4,2]
   + 0.2110073268413544 Conv_1_Relu_ts[4,3]
   - 2.244216442108154 Conv_1_Relu_ts[4,4]
   + 0.1934826672077179 Conv_1_Relu_ts[4,5]
   - 1.220147132873535 Conv_1_Relu_ts[4,6]
   - 0.2545500099658966 Conv_1_Relu_ts[4,7]
   + 1.472832441329956 Conv_1_Relu_ts[4,8]
   - 1.084863781929016 Conv_1_Relu_ts[4,9]
   + 3.159408807754517 Conv_1_Relu_ts[4,10]
   - 1.101905822753906 Conv_1_Relu_ts[4,11]
   + 0.4563636183738708 Conv_1_Relu_ts[4,12]
   - 0.1066205129027367 Conv_1_Relu_ts[4,13]
   + 0.1282547265291214 Conv_1_Relu_ts[4,14]
   - 0.1789750158786774 Conv_1_Relu_ts[4,15]
   + 1.150216221809387 Conv_2_aggregated_features[4,0]
   + 0.7049534916877747 Conv_2_aggregated_features[4,1]
   - 0.6801650524139404 Conv_2_aggregated_features[4,2]
   + 0.0596005544066429 Conv_2_aggregated_features[4,3]
   + 1.205108404159546 Conv_2_aggregated_features[4,4]
   - 0.0892623364925385 Conv_2_aggregated_features[4,5]
   + 1.500906825065613 Conv_2_aggregated_features[4,6]
   - 0.4810487031936646 Conv_2_aggregated_features[4,7]
   + 0.3540807068347931 Conv_2_aggregated_features[4,8]
   - 0.3831521570682526 Conv_2_aggregated_features[4,9]
   - 0.5447250604629517 Conv_2_aggregated_features[4,10]
   + 2.773675441741943 Conv_2_aggregated_features[4,11]
   + 0.4864743649959564 Conv_2_aggregated_features[4,12]
   + 0.2153479158878326 Conv_2_aggregated_features[4,13]
   + 1.067057013511658 Conv_2_aggregated_features[4,14]
   - 0.2962089478969574 Conv_2_aggregated_features[4,15] + Conv_2_t[4,0]
   = -0.1796741187572479
 Conv_2_output_constraint[4,1]: - 0.8503658175468445 Conv_1_Relu_ts[4,0]
   + 0.1598657667636871 Conv_1_Relu_ts[4,1]
   + 0.740400493144989 Conv_1_Relu_ts[4,2]
   - 0.0143840508535504 Conv_1_Relu_ts[4,3]
   + 0.418368011713028 Conv_1_Relu_ts[4,4]
   - 0.0816935896873474 Conv_1_Relu_ts[4,5]
   - 0.6345715522766113 Conv_1_Relu_ts[4,6]
   - 0.1658433526754379 Conv_1_Relu_ts[4,7]
   + 0.2588719427585602 Conv_1_Relu_ts[4,8]
   - 0.3725545704364777 Conv_1_Relu_ts[4,9]
   - 0.1093620359897614 Conv_1_Relu_ts[4,10]
   - 0.229639545083046 Conv_1_Relu_ts[4,11]
   + 2.353754758834839 Conv_1_Relu_ts[4,12]
   + 0.275741845369339 Conv_1_Relu_ts[4,13]
   + 0.2292672693729401 Conv_1_Relu_ts[4,14]
   - 0.085102416574955 Conv_1_Relu_ts[4,15]
   + 1.31701123714447 Conv_2_aggregated_features[4,0]
   + 0.5385994911193848 Conv_2_aggregated_features[4,1]
   - 0.8459239602088928 Conv_2_aggregated_features[4,2]
   + 0.1865081340074539 Conv_2_aggregated_features[4,3]
   + 0.8385969400405884 Conv_2_aggregated_features[4,4]
   - 0.1273856163024902 Conv_2_aggregated_features[4,5]
   + 2.53032374382019 Conv_2_aggregated_features[4,6]
   - 0.3543893694877625 Conv_2_aggregated_features[4,7]
   + 0.1923532038927078 Conv_2_aggregated_features[4,8]
   + 1.131453990936279 Conv_2_aggregated_features[4,9]
   - 0.3651714324951172 Conv_2_aggregated_features[4,10]
   + 1.542536735534668 Conv_2_aggregated_features[4,11]
   - 0.4610330760478973 Conv_2_aggregated_features[4,12]
   - 0.00123762537259609 Conv_2_aggregated_features[4,13]
   + 0.375784307718277 Conv_2_aggregated_features[4,14]
   + 0.0822948440909386 Conv_2_aggregated_features[4,15] + Conv_2_t[4,1]
   = -0.280759334564209
 Conv_2_output_constraint[4,2]: 1.746561527252197 Conv_1_Relu_ts[4,0]
   + 0.814217746257782 Conv_1_Relu_ts[4,1]
   - 1.465980052947998 Conv_1_Relu_ts[4,2]
   - 0.2278070449829102 Conv_1_Relu_ts[4,3]
   + 1.037705659866333 Conv_1_Relu_ts[4,4]
   - 0.2538289427757263 Conv_1_Relu_ts[4,5]
   + 0.6437492966651917 Conv_1_Relu_ts[4,6]
   - 0.4347417056560516 Conv_1_Relu_ts[4,7]
   + 0.2060042917728424 Conv_1_Relu_ts[4,8]
   + 0.8382987976074219 Conv_1_Relu_ts[4,9]
   - 0.6877544522285461 Conv_1_Relu_ts[4,10]
   + 1.350347280502319 Conv_1_Relu_ts[4,11]
   + 2.458039999008179 Conv_1_Relu_ts[4,12]
   - 0.1695821583271027 Conv_1_Relu_ts[4,13]
   + 0.2783661186695099 Conv_1_Relu_ts[4,14]
   - 0.1666328907012939 Conv_1_Relu_ts[4,15]
   - 0.4272331893444061 Conv_2_aggregated_features[4,0]
   + 0.8438286781311035 Conv_2_aggregated_features[4,1]
   - 1.847808957099915 Conv_2_aggregated_features[4,2]
   - 0.1186513304710388 Conv_2_aggregated_features[4,3]
   + 0.6336349844932556 Conv_2_aggregated_features[4,4]
   + 0.0602188408374786 Conv_2_aggregated_features[4,5]
   - 0.2530380189418793 Conv_2_aggregated_features[4,6]
   - 0.5760374069213867 Conv_2_aggregated_features[4,7]
   - 0.8055176734924316 Conv_2_aggregated_features[4,8]
   + 0.4366684854030609 Conv_2_aggregated_features[4,9]
   - 1.177173972129822 Conv_2_aggregated_features[4,10]
   - 0.4784670472145081 Conv_2_aggregated_features[4,11]
   + 2.325191736221313 Conv_2_aggregated_features[4,12]
   - 0.2128628194332123 Conv_2_aggregated_features[4,13]
   - 0.1453570574522018 Conv_2_aggregated_features[4,14]
   - 0.118608683347702 Conv_2_aggregated_features[4,15] + Conv_2_t[4,2]
   = 1.507851004600525
 Conv_2_output_constraint[4,3]: - 0.527165412902832 Conv_1_Relu_ts[4,0]
   - 1.389649629592896 Conv_1_Relu_ts[4,1]
   - 0.6640920639038086 Conv_1_Relu_ts[4,2]
   - 0.0634000822901726 Conv_1_Relu_ts[4,3]
   - 0.4163890480995178 Conv_1_Relu_ts[4,4]
   + 0.0613848567008972 Conv_1_Relu_ts[4,5]
   + 1.790452718734741 Conv_1_Relu_ts[4,6]
   - 0.1128331422805786 Conv_1_Relu_ts[4,7]
   + 2.12717342376709 Conv_1_Relu_ts[4,8]
   - 0.1405593752861023 Conv_1_Relu_ts[4,9]
   - 0.1852237284183502 Conv_1_Relu_ts[4,10]
   - 0.2454802542924881 Conv_1_Relu_ts[4,11]
   - 1.97015655040741 Conv_1_Relu_ts[4,12]
   + 0.3291732370853424 Conv_1_Relu_ts[4,13]
   - 0.5780640244483948 Conv_1_Relu_ts[4,14]
   - 0.1067387536168098 Conv_1_Relu_ts[4,15]
   - 0.1432812511920929 Conv_2_aggregated_features[4,0]
   - 0.4248973727226257 Conv_2_aggregated_features[4,1]
   + 1.125378131866455 Conv_2_aggregated_features[4,2]
   + 0.1260246187448502 Conv_2_aggregated_features[4,3]
   - 0.5538251996040344 Conv_2_aggregated_features[4,4]
   + 0.1312114000320435 Conv_2_aggregated_features[4,5]
   - 0.9671168327331543 Conv_2_aggregated_features[4,6]
   - 0.0926898792386055 Conv_2_aggregated_features[4,7]
   - 0.1030022576451302 Conv_2_aggregated_features[4,8]
   - 1.69473397731781 Conv_2_aggregated_features[4,9]
   + 2.313785552978516 Conv_2_aggregated_features[4,10]
   - 0.3551321625709534 Conv_2_aggregated_features[4,11]
   - 0.441056102514267 Conv_2_aggregated_features[4,12]
   - 0.0765185728669167 Conv_2_aggregated_features[4,13]
   - 0.5983367562294006 Conv_2_aggregated_features[4,14]
   + 0.023831807076931 Conv_2_aggregated_features[4,15] + Conv_2_t[4,3]
   = -0.072962760925293
 Conv_2_output_constraint[4,4]: 0.5219888687133789 Conv_1_Relu_ts[4,0]
   - 1.025602698326111 Conv_1_Relu_ts[4,1]
   - 0.4347750842571259 Conv_1_Relu_ts[4,2]
   + 0.2773195803165436 Conv_1_Relu_ts[4,3]
   - 2.5947265625 Conv_1_Relu_ts[4,4]
   - 0.2330431044101715 Conv_1_Relu_ts[4,5]
   + 0.4634701013565063 Conv_1_Relu_ts[4,6]
   - 0.2591606974601746 Conv_1_Relu_ts[4,7]
   + 0.715503990650177 Conv_1_Relu_ts[4,8]
   - 0.1986619830131531 Conv_1_Relu_ts[4,9]
   - 0.9099530577659607 Conv_1_Relu_ts[4,10]
   - 0.2335229516029358 Conv_1_Relu_ts[4,11]
   - 0.2992779910564423 Conv_1_Relu_ts[4,12]
   + 0.000100307 Conv_1_Relu_ts[4,13]
   - 1.00616979598999 Conv_1_Relu_ts[4,14]
   + 0.2120778858661652 Conv_1_Relu_ts[4,15]
   - 0.0600817315280437 Conv_2_aggregated_features[4,0]
   - 0.099200963973999 Conv_2_aggregated_features[4,1]
   - 0.4625514149665833 Conv_2_aggregated_features[4,2]
   - 0.1394201517105103 Conv_2_aggregated_features[4,3]
   + 1.153588056564331 Conv_2_aggregated_features[4,4]
   - 0.0127062797546387 Conv_2_aggregated_features[4,5]
   + 1.360898375511169 Conv_2_aggregated_features[4,6]
   - 0.1619014590978622 Conv_2_aggregated_features[4,7]
   + 0.0715495049953461 Conv_2_aggregated_features[4,8]
   + 0.1082091927528381 Conv_2_aggregated_features[4,9]
   + 1.219732761383057 Conv_2_aggregated_features[4,10]
   + 0.0949473306536674 Conv_2_aggregated_features[4,11]
   - 0.7561001181602478 Conv_2_aggregated_features[4,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[4,13]
   + 0.3085470497608185 Conv_2_aggregated_features[4,14]
   - 0.1963151693344116 Conv_2_aggregated_features[4,15] + Conv_2_t[4,4]
   = -0.681087851524353
 Conv_2_output_constraint[4,5]: 0.4891856014728546 Conv_1_Relu_ts[4,0]
   - 1.764690637588501 Conv_1_Relu_ts[4,1]
   + 1.572006106376648 Conv_1_Relu_ts[4,2]
   - 0.2603900134563446 Conv_1_Relu_ts[4,3]
   - 2.532854557037354 Conv_1_Relu_ts[4,4]
   - 0.0787733495235443 Conv_1_Relu_ts[4,5]
   + 0.3314015865325928 Conv_1_Relu_ts[4,6]
   + 0.3532952666282654 Conv_1_Relu_ts[4,7]
   + 1.495306849479675 Conv_1_Relu_ts[4,8]
   - 0.3582539260387421 Conv_1_Relu_ts[4,9]
   + 1.239243626594543 Conv_1_Relu_ts[4,10]
   + 0.0796256512403488 Conv_1_Relu_ts[4,11]
   + 1.406325340270996 Conv_1_Relu_ts[4,12]
   + 0.2874834537506104 Conv_1_Relu_ts[4,13]
   - 0.4502094686031342 Conv_1_Relu_ts[4,14]
   - 0.1813227832317352 Conv_1_Relu_ts[4,15]
   - 1.002788901329041 Conv_2_aggregated_features[4,0]
   - 1.268097281455994 Conv_2_aggregated_features[4,1]
   - 1.182242274284363 Conv_2_aggregated_features[4,2]
   - 0.3146973550319672 Conv_2_aggregated_features[4,3]
   + 1.659744143486023 Conv_2_aggregated_features[4,4]
   - 0.0238673985004425 Conv_2_aggregated_features[4,5]
   - 0.0993486419320107 Conv_2_aggregated_features[4,6]
   + 0.1021591797471046 Conv_2_aggregated_features[4,7]
   - 0.9812391996383667 Conv_2_aggregated_features[4,8]
   + 0.6361307501792908 Conv_2_aggregated_features[4,9]
   - 0.2150242179632187 Conv_2_aggregated_features[4,10]
   - 2.391498804092407 Conv_2_aggregated_features[4,11]
   + 0.9408105611801147 Conv_2_aggregated_features[4,12]
   - 0.0835467427968979 Conv_2_aggregated_features[4,13]
   - 0.7100914716720581 Conv_2_aggregated_features[4,14]
   + 0.081059992313385 Conv_2_aggregated_features[4,15] + Conv_2_t[4,5]
   = -1.552430987358093
 Conv_2_output_constraint[4,6]: 0.3276324272155762 Conv_1_Relu_ts[4,0]
   - 1.526854515075684 Conv_1_Relu_ts[4,1]
   - 0.2327080518007278 Conv_1_Relu_ts[4,2]
   + 0.2350454330444336 Conv_1_Relu_ts[4,3]
   - 2.336396932601929 Conv_1_Relu_ts[4,4]
   + 0.0498765707015991 Conv_1_Relu_ts[4,5]
   + 0.1361224502325058 Conv_1_Relu_ts[4,6]
   - 0.7505534887313843 Conv_1_Relu_ts[4,7]
   - 0.1081981733441353 Conv_1_Relu_ts[4,8]
   + 0.36195108294487 Conv_1_Relu_ts[4,9]
   + 0.266346663236618 Conv_1_Relu_ts[4,10]
   - 0.2529528141021729 Conv_1_Relu_ts[4,11]
   + 0.3400895595550537 Conv_1_Relu_ts[4,12]
   - 0.1297148019075394 Conv_1_Relu_ts[4,13]
   + 0.2151342928409576 Conv_1_Relu_ts[4,14]
   - 0.0570834763348103 Conv_1_Relu_ts[4,15]
   - 0.2712872326374054 Conv_2_aggregated_features[4,0]
   - 0.6653857827186584 Conv_2_aggregated_features[4,1]
   + 0.0704844743013382 Conv_2_aggregated_features[4,2]
   - 0.00456566177308559 Conv_2_aggregated_features[4,3]
   + 0.0726276636123657 Conv_2_aggregated_features[4,4]
   - 0.1622210741043091 Conv_2_aggregated_features[4,5]
   - 0.6965416669845581 Conv_2_aggregated_features[4,6]
   - 0.7083758115768433 Conv_2_aggregated_features[4,7]
   - 0.6968990564346313 Conv_2_aggregated_features[4,8]
   + 0.575663149356842 Conv_2_aggregated_features[4,9]
   - 0.3281366527080536 Conv_2_aggregated_features[4,10]
   - 1.37471342086792 Conv_2_aggregated_features[4,11]
   + 1.073476433753967 Conv_2_aggregated_features[4,12]
   + 0.0794747099280357 Conv_2_aggregated_features[4,13]
   + 0.5443784594535828 Conv_2_aggregated_features[4,14]
   - 0.0632283836603165 Conv_2_aggregated_features[4,15] + Conv_2_t[4,6]
   = 0.4907519221305847
 Conv_2_output_constraint[4,7]: - 0.0696223676204681 Conv_1_Relu_ts[4,0]
   - 0.1037913709878922 Conv_1_Relu_ts[4,1]
   + 0.5115458965301514 Conv_1_Relu_ts[4,2]
   - 0.1024313941597939 Conv_1_Relu_ts[4,3]
   + 0.2925211489200592 Conv_1_Relu_ts[4,4]
   + 0.1706541776657104 Conv_1_Relu_ts[4,5]
   + 0.1667102426290512 Conv_1_Relu_ts[4,6]
   + 0.2208153456449509 Conv_1_Relu_ts[4,7]
   + 0.0217245072126389 Conv_1_Relu_ts[4,8]
   + 0.2856383323669434 Conv_1_Relu_ts[4,9]
   + 0.1799796521663666 Conv_1_Relu_ts[4,10]
   + 0.2511521875858307 Conv_1_Relu_ts[4,11]
   + 0.1590938121080399 Conv_1_Relu_ts[4,12]
   - 0.1085780262947083 Conv_1_Relu_ts[4,13]
   - 0.1072387397289276 Conv_1_Relu_ts[4,14]
   + 0.0257974341511726 Conv_1_Relu_ts[4,15]
   - 0.00193481589667499 Conv_2_aggregated_features[4,0]
   - 0.0832197070121765 Conv_2_aggregated_features[4,1]
   + 0.2962732911109924 Conv_2_aggregated_features[4,2]
   + 0.1245143488049507 Conv_2_aggregated_features[4,3]
   + 0.0374857746064663 Conv_2_aggregated_features[4,4]
   - 0.2035515308380127 Conv_2_aggregated_features[4,5]
   + 0.0183010119944811 Conv_2_aggregated_features[4,6]
   - 0.0808482989668846 Conv_2_aggregated_features[4,7]
   + 0.0491516962647438 Conv_2_aggregated_features[4,8]
   - 0.1665307730436325 Conv_2_aggregated_features[4,9]
   + 0.180585503578186 Conv_2_aggregated_features[4,10]
   + 0.0900064706802368 Conv_2_aggregated_features[4,11]
   + 0.1489291191101074 Conv_2_aggregated_features[4,12]
   - 0.1824352592229843 Conv_2_aggregated_features[4,13]
   - 0.0335522592067719 Conv_2_aggregated_features[4,14]
   + 0.1226971745491028 Conv_2_aggregated_features[4,15] + Conv_2_t[4,7]
   = -0.0930573865771294
 Conv_2_output_constraint[5,0]: - 1.283687114715576 Conv_1_Relu_ts[5,0]
   - 0.7330741286277771 Conv_1_Relu_ts[5,1]
   - 2.277222394943237 Conv_1_Relu_ts[5,2]
   + 0.2110073268413544 Conv_1_Relu_ts[5,3]
   - 2.244216442108154 Conv_1_Relu_ts[5,4]
   + 0.1934826672077179 Conv_1_Relu_ts[5,5]
   - 1.220147132873535 Conv_1_Relu_ts[5,6]
   - 0.2545500099658966 Conv_1_Relu_ts[5,7]
   + 1.472832441329956 Conv_1_Relu_ts[5,8]
   - 1.084863781929016 Conv_1_Relu_ts[5,9]
   + 3.159408807754517 Conv_1_Relu_ts[5,10]
   - 1.101905822753906 Conv_1_Relu_ts[5,11]
   + 0.4563636183738708 Conv_1_Relu_ts[5,12]
   - 0.1066205129027367 Conv_1_Relu_ts[5,13]
   + 0.1282547265291214 Conv_1_Relu_ts[5,14]
   - 0.1789750158786774 Conv_1_Relu_ts[5,15]
   + 1.150216221809387 Conv_2_aggregated_features[5,0]
   + 0.7049534916877747 Conv_2_aggregated_features[5,1]
   - 0.6801650524139404 Conv_2_aggregated_features[5,2]
   + 0.0596005544066429 Conv_2_aggregated_features[5,3]
   + 1.205108404159546 Conv_2_aggregated_features[5,4]
   - 0.0892623364925385 Conv_2_aggregated_features[5,5]
   + 1.500906825065613 Conv_2_aggregated_features[5,6]
   - 0.4810487031936646 Conv_2_aggregated_features[5,7]
   + 0.3540807068347931 Conv_2_aggregated_features[5,8]
   - 0.3831521570682526 Conv_2_aggregated_features[5,9]
   - 0.5447250604629517 Conv_2_aggregated_features[5,10]
   + 2.773675441741943 Conv_2_aggregated_features[5,11]
   + 0.4864743649959564 Conv_2_aggregated_features[5,12]
   + 0.2153479158878326 Conv_2_aggregated_features[5,13]
   + 1.067057013511658 Conv_2_aggregated_features[5,14]
   - 0.2962089478969574 Conv_2_aggregated_features[5,15] + Conv_2_t[5,0]
   = -0.1796741187572479
 Conv_2_output_constraint[5,1]: - 0.8503658175468445 Conv_1_Relu_ts[5,0]
   + 0.1598657667636871 Conv_1_Relu_ts[5,1]
   + 0.740400493144989 Conv_1_Relu_ts[5,2]
   - 0.0143840508535504 Conv_1_Relu_ts[5,3]
   + 0.418368011713028 Conv_1_Relu_ts[5,4]
   - 0.0816935896873474 Conv_1_Relu_ts[5,5]
   - 0.6345715522766113 Conv_1_Relu_ts[5,6]
   - 0.1658433526754379 Conv_1_Relu_ts[5,7]
   + 0.2588719427585602 Conv_1_Relu_ts[5,8]
   - 0.3725545704364777 Conv_1_Relu_ts[5,9]
   - 0.1093620359897614 Conv_1_Relu_ts[5,10]
   - 0.229639545083046 Conv_1_Relu_ts[5,11]
   + 2.353754758834839 Conv_1_Relu_ts[5,12]
   + 0.275741845369339 Conv_1_Relu_ts[5,13]
   + 0.2292672693729401 Conv_1_Relu_ts[5,14]
   - 0.085102416574955 Conv_1_Relu_ts[5,15]
   + 1.31701123714447 Conv_2_aggregated_features[5,0]
   + 0.5385994911193848 Conv_2_aggregated_features[5,1]
   - 0.8459239602088928 Conv_2_aggregated_features[5,2]
   + 0.1865081340074539 Conv_2_aggregated_features[5,3]
   + 0.8385969400405884 Conv_2_aggregated_features[5,4]
   - 0.1273856163024902 Conv_2_aggregated_features[5,5]
   + 2.53032374382019 Conv_2_aggregated_features[5,6]
   - 0.3543893694877625 Conv_2_aggregated_features[5,7]
   + 0.1923532038927078 Conv_2_aggregated_features[5,8]
   + 1.131453990936279 Conv_2_aggregated_features[5,9]
   - 0.3651714324951172 Conv_2_aggregated_features[5,10]
   + 1.542536735534668 Conv_2_aggregated_features[5,11]
   - 0.4610330760478973 Conv_2_aggregated_features[5,12]
   - 0.00123762537259609 Conv_2_aggregated_features[5,13]
   + 0.375784307718277 Conv_2_aggregated_features[5,14]
   + 0.0822948440909386 Conv_2_aggregated_features[5,15] + Conv_2_t[5,1]
   = -0.280759334564209
 Conv_2_output_constraint[5,2]: 1.746561527252197 Conv_1_Relu_ts[5,0]
   + 0.814217746257782 Conv_1_Relu_ts[5,1]
   - 1.465980052947998 Conv_1_Relu_ts[5,2]
   - 0.2278070449829102 Conv_1_Relu_ts[5,3]
   + 1.037705659866333 Conv_1_Relu_ts[5,4]
   - 0.2538289427757263 Conv_1_Relu_ts[5,5]
   + 0.6437492966651917 Conv_1_Relu_ts[5,6]
   - 0.4347417056560516 Conv_1_Relu_ts[5,7]
   + 0.2060042917728424 Conv_1_Relu_ts[5,8]
   + 0.8382987976074219 Conv_1_Relu_ts[5,9]
   - 0.6877544522285461 Conv_1_Relu_ts[5,10]
   + 1.350347280502319 Conv_1_Relu_ts[5,11]
   + 2.458039999008179 Conv_1_Relu_ts[5,12]
   - 0.1695821583271027 Conv_1_Relu_ts[5,13]
   + 0.2783661186695099 Conv_1_Relu_ts[5,14]
   - 0.1666328907012939 Conv_1_Relu_ts[5,15]
   - 0.4272331893444061 Conv_2_aggregated_features[5,0]
   + 0.8438286781311035 Conv_2_aggregated_features[5,1]
   - 1.847808957099915 Conv_2_aggregated_features[5,2]
   - 0.1186513304710388 Conv_2_aggregated_features[5,3]
   + 0.6336349844932556 Conv_2_aggregated_features[5,4]
   + 0.0602188408374786 Conv_2_aggregated_features[5,5]
   - 0.2530380189418793 Conv_2_aggregated_features[5,6]
   - 0.5760374069213867 Conv_2_aggregated_features[5,7]
   - 0.8055176734924316 Conv_2_aggregated_features[5,8]
   + 0.4366684854030609 Conv_2_aggregated_features[5,9]
   - 1.177173972129822 Conv_2_aggregated_features[5,10]
   - 0.4784670472145081 Conv_2_aggregated_features[5,11]
   + 2.325191736221313 Conv_2_aggregated_features[5,12]
   - 0.2128628194332123 Conv_2_aggregated_features[5,13]
   - 0.1453570574522018 Conv_2_aggregated_features[5,14]
   - 0.118608683347702 Conv_2_aggregated_features[5,15] + Conv_2_t[5,2]
   = 1.507851004600525
 Conv_2_output_constraint[5,3]: - 0.527165412902832 Conv_1_Relu_ts[5,0]
   - 1.389649629592896 Conv_1_Relu_ts[5,1]
   - 0.6640920639038086 Conv_1_Relu_ts[5,2]
   - 0.0634000822901726 Conv_1_Relu_ts[5,3]
   - 0.4163890480995178 Conv_1_Relu_ts[5,4]
   + 0.0613848567008972 Conv_1_Relu_ts[5,5]
   + 1.790452718734741 Conv_1_Relu_ts[5,6]
   - 0.1128331422805786 Conv_1_Relu_ts[5,7]
   + 2.12717342376709 Conv_1_Relu_ts[5,8]
   - 0.1405593752861023 Conv_1_Relu_ts[5,9]
   - 0.1852237284183502 Conv_1_Relu_ts[5,10]
   - 0.2454802542924881 Conv_1_Relu_ts[5,11]
   - 1.97015655040741 Conv_1_Relu_ts[5,12]
   + 0.3291732370853424 Conv_1_Relu_ts[5,13]
   - 0.5780640244483948 Conv_1_Relu_ts[5,14]
   - 0.1067387536168098 Conv_1_Relu_ts[5,15]
   - 0.1432812511920929 Conv_2_aggregated_features[5,0]
   - 0.4248973727226257 Conv_2_aggregated_features[5,1]
   + 1.125378131866455 Conv_2_aggregated_features[5,2]
   + 0.1260246187448502 Conv_2_aggregated_features[5,3]
   - 0.5538251996040344 Conv_2_aggregated_features[5,4]
   + 0.1312114000320435 Conv_2_aggregated_features[5,5]
   - 0.9671168327331543 Conv_2_aggregated_features[5,6]
   - 0.0926898792386055 Conv_2_aggregated_features[5,7]
   - 0.1030022576451302 Conv_2_aggregated_features[5,8]
   - 1.69473397731781 Conv_2_aggregated_features[5,9]
   + 2.313785552978516 Conv_2_aggregated_features[5,10]
   - 0.3551321625709534 Conv_2_aggregated_features[5,11]
   - 0.441056102514267 Conv_2_aggregated_features[5,12]
   - 0.0765185728669167 Conv_2_aggregated_features[5,13]
   - 0.5983367562294006 Conv_2_aggregated_features[5,14]
   + 0.023831807076931 Conv_2_aggregated_features[5,15] + Conv_2_t[5,3]
   = -0.072962760925293
 Conv_2_output_constraint[5,4]: 0.5219888687133789 Conv_1_Relu_ts[5,0]
   - 1.025602698326111 Conv_1_Relu_ts[5,1]
   - 0.4347750842571259 Conv_1_Relu_ts[5,2]
   + 0.2773195803165436 Conv_1_Relu_ts[5,3]
   - 2.5947265625 Conv_1_Relu_ts[5,4]
   - 0.2330431044101715 Conv_1_Relu_ts[5,5]
   + 0.4634701013565063 Conv_1_Relu_ts[5,6]
   - 0.2591606974601746 Conv_1_Relu_ts[5,7]
   + 0.715503990650177 Conv_1_Relu_ts[5,8]
   - 0.1986619830131531 Conv_1_Relu_ts[5,9]
   - 0.9099530577659607 Conv_1_Relu_ts[5,10]
   - 0.2335229516029358 Conv_1_Relu_ts[5,11]
   - 0.2992779910564423 Conv_1_Relu_ts[5,12]
   + 0.000100307 Conv_1_Relu_ts[5,13]
   - 1.00616979598999 Conv_1_Relu_ts[5,14]
   + 0.2120778858661652 Conv_1_Relu_ts[5,15]
   - 0.0600817315280437 Conv_2_aggregated_features[5,0]
   - 0.099200963973999 Conv_2_aggregated_features[5,1]
   - 0.4625514149665833 Conv_2_aggregated_features[5,2]
   - 0.1394201517105103 Conv_2_aggregated_features[5,3]
   + 1.153588056564331 Conv_2_aggregated_features[5,4]
   - 0.0127062797546387 Conv_2_aggregated_features[5,5]
   + 1.360898375511169 Conv_2_aggregated_features[5,6]
   - 0.1619014590978622 Conv_2_aggregated_features[5,7]
   + 0.0715495049953461 Conv_2_aggregated_features[5,8]
   + 0.1082091927528381 Conv_2_aggregated_features[5,9]
   + 1.219732761383057 Conv_2_aggregated_features[5,10]
   + 0.0949473306536674 Conv_2_aggregated_features[5,11]
   - 0.7561001181602478 Conv_2_aggregated_features[5,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[5,13]
   + 0.3085470497608185 Conv_2_aggregated_features[5,14]
   - 0.1963151693344116 Conv_2_aggregated_features[5,15] + Conv_2_t[5,4]
   = -0.681087851524353
 Conv_2_output_constraint[5,5]: 0.4891856014728546 Conv_1_Relu_ts[5,0]
   - 1.764690637588501 Conv_1_Relu_ts[5,1]
   + 1.572006106376648 Conv_1_Relu_ts[5,2]
   - 0.2603900134563446 Conv_1_Relu_ts[5,3]
   - 2.532854557037354 Conv_1_Relu_ts[5,4]
   - 0.0787733495235443 Conv_1_Relu_ts[5,5]
   + 0.3314015865325928 Conv_1_Relu_ts[5,6]
   + 0.3532952666282654 Conv_1_Relu_ts[5,7]
   + 1.495306849479675 Conv_1_Relu_ts[5,8]
   - 0.3582539260387421 Conv_1_Relu_ts[5,9]
   + 1.239243626594543 Conv_1_Relu_ts[5,10]
   + 0.0796256512403488 Conv_1_Relu_ts[5,11]
   + 1.406325340270996 Conv_1_Relu_ts[5,12]
   + 0.2874834537506104 Conv_1_Relu_ts[5,13]
   - 0.4502094686031342 Conv_1_Relu_ts[5,14]
   - 0.1813227832317352 Conv_1_Relu_ts[5,15]
   - 1.002788901329041 Conv_2_aggregated_features[5,0]
   - 1.268097281455994 Conv_2_aggregated_features[5,1]
   - 1.182242274284363 Conv_2_aggregated_features[5,2]
   - 0.3146973550319672 Conv_2_aggregated_features[5,3]
   + 1.659744143486023 Conv_2_aggregated_features[5,4]
   - 0.0238673985004425 Conv_2_aggregated_features[5,5]
   - 0.0993486419320107 Conv_2_aggregated_features[5,6]
   + 0.1021591797471046 Conv_2_aggregated_features[5,7]
   - 0.9812391996383667 Conv_2_aggregated_features[5,8]
   + 0.6361307501792908 Conv_2_aggregated_features[5,9]
   - 0.2150242179632187 Conv_2_aggregated_features[5,10]
   - 2.391498804092407 Conv_2_aggregated_features[5,11]
   + 0.9408105611801147 Conv_2_aggregated_features[5,12]
   - 0.0835467427968979 Conv_2_aggregated_features[5,13]
   - 0.7100914716720581 Conv_2_aggregated_features[5,14]
   + 0.081059992313385 Conv_2_aggregated_features[5,15] + Conv_2_t[5,5]
   = -1.552430987358093
 Conv_2_output_constraint[5,6]: 0.3276324272155762 Conv_1_Relu_ts[5,0]
   - 1.526854515075684 Conv_1_Relu_ts[5,1]
   - 0.2327080518007278 Conv_1_Relu_ts[5,2]
   + 0.2350454330444336 Conv_1_Relu_ts[5,3]
   - 2.336396932601929 Conv_1_Relu_ts[5,4]
   + 0.0498765707015991 Conv_1_Relu_ts[5,5]
   + 0.1361224502325058 Conv_1_Relu_ts[5,6]
   - 0.7505534887313843 Conv_1_Relu_ts[5,7]
   - 0.1081981733441353 Conv_1_Relu_ts[5,8]
   + 0.36195108294487 Conv_1_Relu_ts[5,9]
   + 0.266346663236618 Conv_1_Relu_ts[5,10]
   - 0.2529528141021729 Conv_1_Relu_ts[5,11]
   + 0.3400895595550537 Conv_1_Relu_ts[5,12]
   - 0.1297148019075394 Conv_1_Relu_ts[5,13]
   + 0.2151342928409576 Conv_1_Relu_ts[5,14]
   - 0.0570834763348103 Conv_1_Relu_ts[5,15]
   - 0.2712872326374054 Conv_2_aggregated_features[5,0]
   - 0.6653857827186584 Conv_2_aggregated_features[5,1]
   + 0.0704844743013382 Conv_2_aggregated_features[5,2]
   - 0.00456566177308559 Conv_2_aggregated_features[5,3]
   + 0.0726276636123657 Conv_2_aggregated_features[5,4]
   - 0.1622210741043091 Conv_2_aggregated_features[5,5]
   - 0.6965416669845581 Conv_2_aggregated_features[5,6]
   - 0.7083758115768433 Conv_2_aggregated_features[5,7]
   - 0.6968990564346313 Conv_2_aggregated_features[5,8]
   + 0.575663149356842 Conv_2_aggregated_features[5,9]
   - 0.3281366527080536 Conv_2_aggregated_features[5,10]
   - 1.37471342086792 Conv_2_aggregated_features[5,11]
   + 1.073476433753967 Conv_2_aggregated_features[5,12]
   + 0.0794747099280357 Conv_2_aggregated_features[5,13]
   + 0.5443784594535828 Conv_2_aggregated_features[5,14]
   - 0.0632283836603165 Conv_2_aggregated_features[5,15] + Conv_2_t[5,6]
   = 0.4907519221305847
 Conv_2_output_constraint[5,7]: - 0.0696223676204681 Conv_1_Relu_ts[5,0]
   - 0.1037913709878922 Conv_1_Relu_ts[5,1]
   + 0.5115458965301514 Conv_1_Relu_ts[5,2]
   - 0.1024313941597939 Conv_1_Relu_ts[5,3]
   + 0.2925211489200592 Conv_1_Relu_ts[5,4]
   + 0.1706541776657104 Conv_1_Relu_ts[5,5]
   + 0.1667102426290512 Conv_1_Relu_ts[5,6]
   + 0.2208153456449509 Conv_1_Relu_ts[5,7]
   + 0.0217245072126389 Conv_1_Relu_ts[5,8]
   + 0.2856383323669434 Conv_1_Relu_ts[5,9]
   + 0.1799796521663666 Conv_1_Relu_ts[5,10]
   + 0.2511521875858307 Conv_1_Relu_ts[5,11]
   + 0.1590938121080399 Conv_1_Relu_ts[5,12]
   - 0.1085780262947083 Conv_1_Relu_ts[5,13]
   - 0.1072387397289276 Conv_1_Relu_ts[5,14]
   + 0.0257974341511726 Conv_1_Relu_ts[5,15]
   - 0.00193481589667499 Conv_2_aggregated_features[5,0]
   - 0.0832197070121765 Conv_2_aggregated_features[5,1]
   + 0.2962732911109924 Conv_2_aggregated_features[5,2]
   + 0.1245143488049507 Conv_2_aggregated_features[5,3]
   + 0.0374857746064663 Conv_2_aggregated_features[5,4]
   - 0.2035515308380127 Conv_2_aggregated_features[5,5]
   + 0.0183010119944811 Conv_2_aggregated_features[5,6]
   - 0.0808482989668846 Conv_2_aggregated_features[5,7]
   + 0.0491516962647438 Conv_2_aggregated_features[5,8]
   - 0.1665307730436325 Conv_2_aggregated_features[5,9]
   + 0.180585503578186 Conv_2_aggregated_features[5,10]
   + 0.0900064706802368 Conv_2_aggregated_features[5,11]
   + 0.1489291191101074 Conv_2_aggregated_features[5,12]
   - 0.1824352592229843 Conv_2_aggregated_features[5,13]
   - 0.0335522592067719 Conv_2_aggregated_features[5,14]
   + 0.1226971745491028 Conv_2_aggregated_features[5,15] + Conv_2_t[5,7]
   = -0.0930573865771294
 Conv_2_output_constraint[6,0]: - 1.283687114715576 Conv_1_Relu_ts[6,0]
   - 0.7330741286277771 Conv_1_Relu_ts[6,1]
   - 2.277222394943237 Conv_1_Relu_ts[6,2]
   + 0.2110073268413544 Conv_1_Relu_ts[6,3]
   - 2.244216442108154 Conv_1_Relu_ts[6,4]
   + 0.1934826672077179 Conv_1_Relu_ts[6,5]
   - 1.220147132873535 Conv_1_Relu_ts[6,6]
   - 0.2545500099658966 Conv_1_Relu_ts[6,7]
   + 1.472832441329956 Conv_1_Relu_ts[6,8]
   - 1.084863781929016 Conv_1_Relu_ts[6,9]
   + 3.159408807754517 Conv_1_Relu_ts[6,10]
   - 1.101905822753906 Conv_1_Relu_ts[6,11]
   + 0.4563636183738708 Conv_1_Relu_ts[6,12]
   - 0.1066205129027367 Conv_1_Relu_ts[6,13]
   + 0.1282547265291214 Conv_1_Relu_ts[6,14]
   - 0.1789750158786774 Conv_1_Relu_ts[6,15]
   + 1.150216221809387 Conv_2_aggregated_features[6,0]
   + 0.7049534916877747 Conv_2_aggregated_features[6,1]
   - 0.6801650524139404 Conv_2_aggregated_features[6,2]
   + 0.0596005544066429 Conv_2_aggregated_features[6,3]
   + 1.205108404159546 Conv_2_aggregated_features[6,4]
   - 0.0892623364925385 Conv_2_aggregated_features[6,5]
   + 1.500906825065613 Conv_2_aggregated_features[6,6]
   - 0.4810487031936646 Conv_2_aggregated_features[6,7]
   + 0.3540807068347931 Conv_2_aggregated_features[6,8]
   - 0.3831521570682526 Conv_2_aggregated_features[6,9]
   - 0.5447250604629517 Conv_2_aggregated_features[6,10]
   + 2.773675441741943 Conv_2_aggregated_features[6,11]
   + 0.4864743649959564 Conv_2_aggregated_features[6,12]
   + 0.2153479158878326 Conv_2_aggregated_features[6,13]
   + 1.067057013511658 Conv_2_aggregated_features[6,14]
   - 0.2962089478969574 Conv_2_aggregated_features[6,15] + Conv_2_t[6,0]
   = -0.1796741187572479
 Conv_2_output_constraint[6,1]: - 0.8503658175468445 Conv_1_Relu_ts[6,0]
   + 0.1598657667636871 Conv_1_Relu_ts[6,1]
   + 0.740400493144989 Conv_1_Relu_ts[6,2]
   - 0.0143840508535504 Conv_1_Relu_ts[6,3]
   + 0.418368011713028 Conv_1_Relu_ts[6,4]
   - 0.0816935896873474 Conv_1_Relu_ts[6,5]
   - 0.6345715522766113 Conv_1_Relu_ts[6,6]
   - 0.1658433526754379 Conv_1_Relu_ts[6,7]
   + 0.2588719427585602 Conv_1_Relu_ts[6,8]
   - 0.3725545704364777 Conv_1_Relu_ts[6,9]
   - 0.1093620359897614 Conv_1_Relu_ts[6,10]
   - 0.229639545083046 Conv_1_Relu_ts[6,11]
   + 2.353754758834839 Conv_1_Relu_ts[6,12]
   + 0.275741845369339 Conv_1_Relu_ts[6,13]
   + 0.2292672693729401 Conv_1_Relu_ts[6,14]
   - 0.085102416574955 Conv_1_Relu_ts[6,15]
   + 1.31701123714447 Conv_2_aggregated_features[6,0]
   + 0.5385994911193848 Conv_2_aggregated_features[6,1]
   - 0.8459239602088928 Conv_2_aggregated_features[6,2]
   + 0.1865081340074539 Conv_2_aggregated_features[6,3]
   + 0.8385969400405884 Conv_2_aggregated_features[6,4]
   - 0.1273856163024902 Conv_2_aggregated_features[6,5]
   + 2.53032374382019 Conv_2_aggregated_features[6,6]
   - 0.3543893694877625 Conv_2_aggregated_features[6,7]
   + 0.1923532038927078 Conv_2_aggregated_features[6,8]
   + 1.131453990936279 Conv_2_aggregated_features[6,9]
   - 0.3651714324951172 Conv_2_aggregated_features[6,10]
   + 1.542536735534668 Conv_2_aggregated_features[6,11]
   - 0.4610330760478973 Conv_2_aggregated_features[6,12]
   - 0.00123762537259609 Conv_2_aggregated_features[6,13]
   + 0.375784307718277 Conv_2_aggregated_features[6,14]
   + 0.0822948440909386 Conv_2_aggregated_features[6,15] + Conv_2_t[6,1]
   = -0.280759334564209
 Conv_2_output_constraint[6,2]: 1.746561527252197 Conv_1_Relu_ts[6,0]
   + 0.814217746257782 Conv_1_Relu_ts[6,1]
   - 1.465980052947998 Conv_1_Relu_ts[6,2]
   - 0.2278070449829102 Conv_1_Relu_ts[6,3]
   + 1.037705659866333 Conv_1_Relu_ts[6,4]
   - 0.2538289427757263 Conv_1_Relu_ts[6,5]
   + 0.6437492966651917 Conv_1_Relu_ts[6,6]
   - 0.4347417056560516 Conv_1_Relu_ts[6,7]
   + 0.2060042917728424 Conv_1_Relu_ts[6,8]
   + 0.8382987976074219 Conv_1_Relu_ts[6,9]
   - 0.6877544522285461 Conv_1_Relu_ts[6,10]
   + 1.350347280502319 Conv_1_Relu_ts[6,11]
   + 2.458039999008179 Conv_1_Relu_ts[6,12]
   - 0.1695821583271027 Conv_1_Relu_ts[6,13]
   + 0.2783661186695099 Conv_1_Relu_ts[6,14]
   - 0.1666328907012939 Conv_1_Relu_ts[6,15]
   - 0.4272331893444061 Conv_2_aggregated_features[6,0]
   + 0.8438286781311035 Conv_2_aggregated_features[6,1]
   - 1.847808957099915 Conv_2_aggregated_features[6,2]
   - 0.1186513304710388 Conv_2_aggregated_features[6,3]
   + 0.6336349844932556 Conv_2_aggregated_features[6,4]
   + 0.0602188408374786 Conv_2_aggregated_features[6,5]
   - 0.2530380189418793 Conv_2_aggregated_features[6,6]
   - 0.5760374069213867 Conv_2_aggregated_features[6,7]
   - 0.8055176734924316 Conv_2_aggregated_features[6,8]
   + 0.4366684854030609 Conv_2_aggregated_features[6,9]
   - 1.177173972129822 Conv_2_aggregated_features[6,10]
   - 0.4784670472145081 Conv_2_aggregated_features[6,11]
   + 2.325191736221313 Conv_2_aggregated_features[6,12]
   - 0.2128628194332123 Conv_2_aggregated_features[6,13]
   - 0.1453570574522018 Conv_2_aggregated_features[6,14]
   - 0.118608683347702 Conv_2_aggregated_features[6,15] + Conv_2_t[6,2]
   = 1.507851004600525
 Conv_2_output_constraint[6,3]: - 0.527165412902832 Conv_1_Relu_ts[6,0]
   - 1.389649629592896 Conv_1_Relu_ts[6,1]
   - 0.6640920639038086 Conv_1_Relu_ts[6,2]
   - 0.0634000822901726 Conv_1_Relu_ts[6,3]
   - 0.4163890480995178 Conv_1_Relu_ts[6,4]
   + 0.0613848567008972 Conv_1_Relu_ts[6,5]
   + 1.790452718734741 Conv_1_Relu_ts[6,6]
   - 0.1128331422805786 Conv_1_Relu_ts[6,7]
   + 2.12717342376709 Conv_1_Relu_ts[6,8]
   - 0.1405593752861023 Conv_1_Relu_ts[6,9]
   - 0.1852237284183502 Conv_1_Relu_ts[6,10]
   - 0.2454802542924881 Conv_1_Relu_ts[6,11]
   - 1.97015655040741 Conv_1_Relu_ts[6,12]
   + 0.3291732370853424 Conv_1_Relu_ts[6,13]
   - 0.5780640244483948 Conv_1_Relu_ts[6,14]
   - 0.1067387536168098 Conv_1_Relu_ts[6,15]
   - 0.1432812511920929 Conv_2_aggregated_features[6,0]
   - 0.4248973727226257 Conv_2_aggregated_features[6,1]
   + 1.125378131866455 Conv_2_aggregated_features[6,2]
   + 0.1260246187448502 Conv_2_aggregated_features[6,3]
   - 0.5538251996040344 Conv_2_aggregated_features[6,4]
   + 0.1312114000320435 Conv_2_aggregated_features[6,5]
   - 0.9671168327331543 Conv_2_aggregated_features[6,6]
   - 0.0926898792386055 Conv_2_aggregated_features[6,7]
   - 0.1030022576451302 Conv_2_aggregated_features[6,8]
   - 1.69473397731781 Conv_2_aggregated_features[6,9]
   + 2.313785552978516 Conv_2_aggregated_features[6,10]
   - 0.3551321625709534 Conv_2_aggregated_features[6,11]
   - 0.441056102514267 Conv_2_aggregated_features[6,12]
   - 0.0765185728669167 Conv_2_aggregated_features[6,13]
   - 0.5983367562294006 Conv_2_aggregated_features[6,14]
   + 0.023831807076931 Conv_2_aggregated_features[6,15] + Conv_2_t[6,3]
   = -0.072962760925293
 Conv_2_output_constraint[6,4]: 0.5219888687133789 Conv_1_Relu_ts[6,0]
   - 1.025602698326111 Conv_1_Relu_ts[6,1]
   - 0.4347750842571259 Conv_1_Relu_ts[6,2]
   + 0.2773195803165436 Conv_1_Relu_ts[6,3]
   - 2.5947265625 Conv_1_Relu_ts[6,4]
   - 0.2330431044101715 Conv_1_Relu_ts[6,5]
   + 0.4634701013565063 Conv_1_Relu_ts[6,6]
   - 0.2591606974601746 Conv_1_Relu_ts[6,7]
   + 0.715503990650177 Conv_1_Relu_ts[6,8]
   - 0.1986619830131531 Conv_1_Relu_ts[6,9]
   - 0.9099530577659607 Conv_1_Relu_ts[6,10]
   - 0.2335229516029358 Conv_1_Relu_ts[6,11]
   - 0.2992779910564423 Conv_1_Relu_ts[6,12]
   + 0.000100307 Conv_1_Relu_ts[6,13]
   - 1.00616979598999 Conv_1_Relu_ts[6,14]
   + 0.2120778858661652 Conv_1_Relu_ts[6,15]
   - 0.0600817315280437 Conv_2_aggregated_features[6,0]
   - 0.099200963973999 Conv_2_aggregated_features[6,1]
   - 0.4625514149665833 Conv_2_aggregated_features[6,2]
   - 0.1394201517105103 Conv_2_aggregated_features[6,3]
   + 1.153588056564331 Conv_2_aggregated_features[6,4]
   - 0.0127062797546387 Conv_2_aggregated_features[6,5]
   + 1.360898375511169 Conv_2_aggregated_features[6,6]
   - 0.1619014590978622 Conv_2_aggregated_features[6,7]
   + 0.0715495049953461 Conv_2_aggregated_features[6,8]
   + 0.1082091927528381 Conv_2_aggregated_features[6,9]
   + 1.219732761383057 Conv_2_aggregated_features[6,10]
   + 0.0949473306536674 Conv_2_aggregated_features[6,11]
   - 0.7561001181602478 Conv_2_aggregated_features[6,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[6,13]
   + 0.3085470497608185 Conv_2_aggregated_features[6,14]
   - 0.1963151693344116 Conv_2_aggregated_features[6,15] + Conv_2_t[6,4]
   = -0.681087851524353
 Conv_2_output_constraint[6,5]: 0.4891856014728546 Conv_1_Relu_ts[6,0]
   - 1.764690637588501 Conv_1_Relu_ts[6,1]
   + 1.572006106376648 Conv_1_Relu_ts[6,2]
   - 0.2603900134563446 Conv_1_Relu_ts[6,3]
   - 2.532854557037354 Conv_1_Relu_ts[6,4]
   - 0.0787733495235443 Conv_1_Relu_ts[6,5]
   + 0.3314015865325928 Conv_1_Relu_ts[6,6]
   + 0.3532952666282654 Conv_1_Relu_ts[6,7]
   + 1.495306849479675 Conv_1_Relu_ts[6,8]
   - 0.3582539260387421 Conv_1_Relu_ts[6,9]
   + 1.239243626594543 Conv_1_Relu_ts[6,10]
   + 0.0796256512403488 Conv_1_Relu_ts[6,11]
   + 1.406325340270996 Conv_1_Relu_ts[6,12]
   + 0.2874834537506104 Conv_1_Relu_ts[6,13]
   - 0.4502094686031342 Conv_1_Relu_ts[6,14]
   - 0.1813227832317352 Conv_1_Relu_ts[6,15]
   - 1.002788901329041 Conv_2_aggregated_features[6,0]
   - 1.268097281455994 Conv_2_aggregated_features[6,1]
   - 1.182242274284363 Conv_2_aggregated_features[6,2]
   - 0.3146973550319672 Conv_2_aggregated_features[6,3]
   + 1.659744143486023 Conv_2_aggregated_features[6,4]
   - 0.0238673985004425 Conv_2_aggregated_features[6,5]
   - 0.0993486419320107 Conv_2_aggregated_features[6,6]
   + 0.1021591797471046 Conv_2_aggregated_features[6,7]
   - 0.9812391996383667 Conv_2_aggregated_features[6,8]
   + 0.6361307501792908 Conv_2_aggregated_features[6,9]
   - 0.2150242179632187 Conv_2_aggregated_features[6,10]
   - 2.391498804092407 Conv_2_aggregated_features[6,11]
   + 0.9408105611801147 Conv_2_aggregated_features[6,12]
   - 0.0835467427968979 Conv_2_aggregated_features[6,13]
   - 0.7100914716720581 Conv_2_aggregated_features[6,14]
   + 0.081059992313385 Conv_2_aggregated_features[6,15] + Conv_2_t[6,5]
   = -1.552430987358093
 Conv_2_output_constraint[6,6]: 0.3276324272155762 Conv_1_Relu_ts[6,0]
   - 1.526854515075684 Conv_1_Relu_ts[6,1]
   - 0.2327080518007278 Conv_1_Relu_ts[6,2]
   + 0.2350454330444336 Conv_1_Relu_ts[6,3]
   - 2.336396932601929 Conv_1_Relu_ts[6,4]
   + 0.0498765707015991 Conv_1_Relu_ts[6,5]
   + 0.1361224502325058 Conv_1_Relu_ts[6,6]
   - 0.7505534887313843 Conv_1_Relu_ts[6,7]
   - 0.1081981733441353 Conv_1_Relu_ts[6,8]
   + 0.36195108294487 Conv_1_Relu_ts[6,9]
   + 0.266346663236618 Conv_1_Relu_ts[6,10]
   - 0.2529528141021729 Conv_1_Relu_ts[6,11]
   + 0.3400895595550537 Conv_1_Relu_ts[6,12]
   - 0.1297148019075394 Conv_1_Relu_ts[6,13]
   + 0.2151342928409576 Conv_1_Relu_ts[6,14]
   - 0.0570834763348103 Conv_1_Relu_ts[6,15]
   - 0.2712872326374054 Conv_2_aggregated_features[6,0]
   - 0.6653857827186584 Conv_2_aggregated_features[6,1]
   + 0.0704844743013382 Conv_2_aggregated_features[6,2]
   - 0.00456566177308559 Conv_2_aggregated_features[6,3]
   + 0.0726276636123657 Conv_2_aggregated_features[6,4]
   - 0.1622210741043091 Conv_2_aggregated_features[6,5]
   - 0.6965416669845581 Conv_2_aggregated_features[6,6]
   - 0.7083758115768433 Conv_2_aggregated_features[6,7]
   - 0.6968990564346313 Conv_2_aggregated_features[6,8]
   + 0.575663149356842 Conv_2_aggregated_features[6,9]
   - 0.3281366527080536 Conv_2_aggregated_features[6,10]
   - 1.37471342086792 Conv_2_aggregated_features[6,11]
   + 1.073476433753967 Conv_2_aggregated_features[6,12]
   + 0.0794747099280357 Conv_2_aggregated_features[6,13]
   + 0.5443784594535828 Conv_2_aggregated_features[6,14]
   - 0.0632283836603165 Conv_2_aggregated_features[6,15] + Conv_2_t[6,6]
   = 0.4907519221305847
 Conv_2_output_constraint[6,7]: - 0.0696223676204681 Conv_1_Relu_ts[6,0]
   - 0.1037913709878922 Conv_1_Relu_ts[6,1]
   + 0.5115458965301514 Conv_1_Relu_ts[6,2]
   - 0.1024313941597939 Conv_1_Relu_ts[6,3]
   + 0.2925211489200592 Conv_1_Relu_ts[6,4]
   + 0.1706541776657104 Conv_1_Relu_ts[6,5]
   + 0.1667102426290512 Conv_1_Relu_ts[6,6]
   + 0.2208153456449509 Conv_1_Relu_ts[6,7]
   + 0.0217245072126389 Conv_1_Relu_ts[6,8]
   + 0.2856383323669434 Conv_1_Relu_ts[6,9]
   + 0.1799796521663666 Conv_1_Relu_ts[6,10]
   + 0.2511521875858307 Conv_1_Relu_ts[6,11]
   + 0.1590938121080399 Conv_1_Relu_ts[6,12]
   - 0.1085780262947083 Conv_1_Relu_ts[6,13]
   - 0.1072387397289276 Conv_1_Relu_ts[6,14]
   + 0.0257974341511726 Conv_1_Relu_ts[6,15]
   - 0.00193481589667499 Conv_2_aggregated_features[6,0]
   - 0.0832197070121765 Conv_2_aggregated_features[6,1]
   + 0.2962732911109924 Conv_2_aggregated_features[6,2]
   + 0.1245143488049507 Conv_2_aggregated_features[6,3]
   + 0.0374857746064663 Conv_2_aggregated_features[6,4]
   - 0.2035515308380127 Conv_2_aggregated_features[6,5]
   + 0.0183010119944811 Conv_2_aggregated_features[6,6]
   - 0.0808482989668846 Conv_2_aggregated_features[6,7]
   + 0.0491516962647438 Conv_2_aggregated_features[6,8]
   - 0.1665307730436325 Conv_2_aggregated_features[6,9]
   + 0.180585503578186 Conv_2_aggregated_features[6,10]
   + 0.0900064706802368 Conv_2_aggregated_features[6,11]
   + 0.1489291191101074 Conv_2_aggregated_features[6,12]
   - 0.1824352592229843 Conv_2_aggregated_features[6,13]
   - 0.0335522592067719 Conv_2_aggregated_features[6,14]
   + 0.1226971745491028 Conv_2_aggregated_features[6,15] + Conv_2_t[6,7]
   = -0.0930573865771294
 Conv_2_output_constraint[7,0]: - 1.283687114715576 Conv_1_Relu_ts[7,0]
   - 0.7330741286277771 Conv_1_Relu_ts[7,1]
   - 2.277222394943237 Conv_1_Relu_ts[7,2]
   + 0.2110073268413544 Conv_1_Relu_ts[7,3]
   - 2.244216442108154 Conv_1_Relu_ts[7,4]
   + 0.1934826672077179 Conv_1_Relu_ts[7,5]
   - 1.220147132873535 Conv_1_Relu_ts[7,6]
   - 0.2545500099658966 Conv_1_Relu_ts[7,7]
   + 1.472832441329956 Conv_1_Relu_ts[7,8]
   - 1.084863781929016 Conv_1_Relu_ts[7,9]
   + 3.159408807754517 Conv_1_Relu_ts[7,10]
   - 1.101905822753906 Conv_1_Relu_ts[7,11]
   + 0.4563636183738708 Conv_1_Relu_ts[7,12]
   - 0.1066205129027367 Conv_1_Relu_ts[7,13]
   + 0.1282547265291214 Conv_1_Relu_ts[7,14]
   - 0.1789750158786774 Conv_1_Relu_ts[7,15]
   + 1.150216221809387 Conv_2_aggregated_features[7,0]
   + 0.7049534916877747 Conv_2_aggregated_features[7,1]
   - 0.6801650524139404 Conv_2_aggregated_features[7,2]
   + 0.0596005544066429 Conv_2_aggregated_features[7,3]
   + 1.205108404159546 Conv_2_aggregated_features[7,4]
   - 0.0892623364925385 Conv_2_aggregated_features[7,5]
   + 1.500906825065613 Conv_2_aggregated_features[7,6]
   - 0.4810487031936646 Conv_2_aggregated_features[7,7]
   + 0.3540807068347931 Conv_2_aggregated_features[7,8]
   - 0.3831521570682526 Conv_2_aggregated_features[7,9]
   - 0.5447250604629517 Conv_2_aggregated_features[7,10]
   + 2.773675441741943 Conv_2_aggregated_features[7,11]
   + 0.4864743649959564 Conv_2_aggregated_features[7,12]
   + 0.2153479158878326 Conv_2_aggregated_features[7,13]
   + 1.067057013511658 Conv_2_aggregated_features[7,14]
   - 0.2962089478969574 Conv_2_aggregated_features[7,15] + Conv_2_t[7,0]
   = -0.1796741187572479
 Conv_2_output_constraint[7,1]: - 0.8503658175468445 Conv_1_Relu_ts[7,0]
   + 0.1598657667636871 Conv_1_Relu_ts[7,1]
   + 0.740400493144989 Conv_1_Relu_ts[7,2]
   - 0.0143840508535504 Conv_1_Relu_ts[7,3]
   + 0.418368011713028 Conv_1_Relu_ts[7,4]
   - 0.0816935896873474 Conv_1_Relu_ts[7,5]
   - 0.6345715522766113 Conv_1_Relu_ts[7,6]
   - 0.1658433526754379 Conv_1_Relu_ts[7,7]
   + 0.2588719427585602 Conv_1_Relu_ts[7,8]
   - 0.3725545704364777 Conv_1_Relu_ts[7,9]
   - 0.1093620359897614 Conv_1_Relu_ts[7,10]
   - 0.229639545083046 Conv_1_Relu_ts[7,11]
   + 2.353754758834839 Conv_1_Relu_ts[7,12]
   + 0.275741845369339 Conv_1_Relu_ts[7,13]
   + 0.2292672693729401 Conv_1_Relu_ts[7,14]
   - 0.085102416574955 Conv_1_Relu_ts[7,15]
   + 1.31701123714447 Conv_2_aggregated_features[7,0]
   + 0.5385994911193848 Conv_2_aggregated_features[7,1]
   - 0.8459239602088928 Conv_2_aggregated_features[7,2]
   + 0.1865081340074539 Conv_2_aggregated_features[7,3]
   + 0.8385969400405884 Conv_2_aggregated_features[7,4]
   - 0.1273856163024902 Conv_2_aggregated_features[7,5]
   + 2.53032374382019 Conv_2_aggregated_features[7,6]
   - 0.3543893694877625 Conv_2_aggregated_features[7,7]
   + 0.1923532038927078 Conv_2_aggregated_features[7,8]
   + 1.131453990936279 Conv_2_aggregated_features[7,9]
   - 0.3651714324951172 Conv_2_aggregated_features[7,10]
   + 1.542536735534668 Conv_2_aggregated_features[7,11]
   - 0.4610330760478973 Conv_2_aggregated_features[7,12]
   - 0.00123762537259609 Conv_2_aggregated_features[7,13]
   + 0.375784307718277 Conv_2_aggregated_features[7,14]
   + 0.0822948440909386 Conv_2_aggregated_features[7,15] + Conv_2_t[7,1]
   = -0.280759334564209
 Conv_2_output_constraint[7,2]: 1.746561527252197 Conv_1_Relu_ts[7,0]
   + 0.814217746257782 Conv_1_Relu_ts[7,1]
   - 1.465980052947998 Conv_1_Relu_ts[7,2]
   - 0.2278070449829102 Conv_1_Relu_ts[7,3]
   + 1.037705659866333 Conv_1_Relu_ts[7,4]
   - 0.2538289427757263 Conv_1_Relu_ts[7,5]
   + 0.6437492966651917 Conv_1_Relu_ts[7,6]
   - 0.4347417056560516 Conv_1_Relu_ts[7,7]
   + 0.2060042917728424 Conv_1_Relu_ts[7,8]
   + 0.8382987976074219 Conv_1_Relu_ts[7,9]
   - 0.6877544522285461 Conv_1_Relu_ts[7,10]
   + 1.350347280502319 Conv_1_Relu_ts[7,11]
   + 2.458039999008179 Conv_1_Relu_ts[7,12]
   - 0.1695821583271027 Conv_1_Relu_ts[7,13]
   + 0.2783661186695099 Conv_1_Relu_ts[7,14]
   - 0.1666328907012939 Conv_1_Relu_ts[7,15]
   - 0.4272331893444061 Conv_2_aggregated_features[7,0]
   + 0.8438286781311035 Conv_2_aggregated_features[7,1]
   - 1.847808957099915 Conv_2_aggregated_features[7,2]
   - 0.1186513304710388 Conv_2_aggregated_features[7,3]
   + 0.6336349844932556 Conv_2_aggregated_features[7,4]
   + 0.0602188408374786 Conv_2_aggregated_features[7,5]
   - 0.2530380189418793 Conv_2_aggregated_features[7,6]
   - 0.5760374069213867 Conv_2_aggregated_features[7,7]
   - 0.8055176734924316 Conv_2_aggregated_features[7,8]
   + 0.4366684854030609 Conv_2_aggregated_features[7,9]
   - 1.177173972129822 Conv_2_aggregated_features[7,10]
   - 0.4784670472145081 Conv_2_aggregated_features[7,11]
   + 2.325191736221313 Conv_2_aggregated_features[7,12]
   - 0.2128628194332123 Conv_2_aggregated_features[7,13]
   - 0.1453570574522018 Conv_2_aggregated_features[7,14]
   - 0.118608683347702 Conv_2_aggregated_features[7,15] + Conv_2_t[7,2]
   = 1.507851004600525
 Conv_2_output_constraint[7,3]: - 0.527165412902832 Conv_1_Relu_ts[7,0]
   - 1.389649629592896 Conv_1_Relu_ts[7,1]
   - 0.6640920639038086 Conv_1_Relu_ts[7,2]
   - 0.0634000822901726 Conv_1_Relu_ts[7,3]
   - 0.4163890480995178 Conv_1_Relu_ts[7,4]
   + 0.0613848567008972 Conv_1_Relu_ts[7,5]
   + 1.790452718734741 Conv_1_Relu_ts[7,6]
   - 0.1128331422805786 Conv_1_Relu_ts[7,7]
   + 2.12717342376709 Conv_1_Relu_ts[7,8]
   - 0.1405593752861023 Conv_1_Relu_ts[7,9]
   - 0.1852237284183502 Conv_1_Relu_ts[7,10]
   - 0.2454802542924881 Conv_1_Relu_ts[7,11]
   - 1.97015655040741 Conv_1_Relu_ts[7,12]
   + 0.3291732370853424 Conv_1_Relu_ts[7,13]
   - 0.5780640244483948 Conv_1_Relu_ts[7,14]
   - 0.1067387536168098 Conv_1_Relu_ts[7,15]
   - 0.1432812511920929 Conv_2_aggregated_features[7,0]
   - 0.4248973727226257 Conv_2_aggregated_features[7,1]
   + 1.125378131866455 Conv_2_aggregated_features[7,2]
   + 0.1260246187448502 Conv_2_aggregated_features[7,3]
   - 0.5538251996040344 Conv_2_aggregated_features[7,4]
   + 0.1312114000320435 Conv_2_aggregated_features[7,5]
   - 0.9671168327331543 Conv_2_aggregated_features[7,6]
   - 0.0926898792386055 Conv_2_aggregated_features[7,7]
   - 0.1030022576451302 Conv_2_aggregated_features[7,8]
   - 1.69473397731781 Conv_2_aggregated_features[7,9]
   + 2.313785552978516 Conv_2_aggregated_features[7,10]
   - 0.3551321625709534 Conv_2_aggregated_features[7,11]
   - 0.441056102514267 Conv_2_aggregated_features[7,12]
   - 0.0765185728669167 Conv_2_aggregated_features[7,13]
   - 0.5983367562294006 Conv_2_aggregated_features[7,14]
   + 0.023831807076931 Conv_2_aggregated_features[7,15] + Conv_2_t[7,3]
   = -0.072962760925293
 Conv_2_output_constraint[7,4]: 0.5219888687133789 Conv_1_Relu_ts[7,0]
   - 1.025602698326111 Conv_1_Relu_ts[7,1]
   - 0.4347750842571259 Conv_1_Relu_ts[7,2]
   + 0.2773195803165436 Conv_1_Relu_ts[7,3]
   - 2.5947265625 Conv_1_Relu_ts[7,4]
   - 0.2330431044101715 Conv_1_Relu_ts[7,5]
   + 0.4634701013565063 Conv_1_Relu_ts[7,6]
   - 0.2591606974601746 Conv_1_Relu_ts[7,7]
   + 0.715503990650177 Conv_1_Relu_ts[7,8]
   - 0.1986619830131531 Conv_1_Relu_ts[7,9]
   - 0.9099530577659607 Conv_1_Relu_ts[7,10]
   - 0.2335229516029358 Conv_1_Relu_ts[7,11]
   - 0.2992779910564423 Conv_1_Relu_ts[7,12]
   + 0.000100307 Conv_1_Relu_ts[7,13]
   - 1.00616979598999 Conv_1_Relu_ts[7,14]
   + 0.2120778858661652 Conv_1_Relu_ts[7,15]
   - 0.0600817315280437 Conv_2_aggregated_features[7,0]
   - 0.099200963973999 Conv_2_aggregated_features[7,1]
   - 0.4625514149665833 Conv_2_aggregated_features[7,2]
   - 0.1394201517105103 Conv_2_aggregated_features[7,3]
   + 1.153588056564331 Conv_2_aggregated_features[7,4]
   - 0.0127062797546387 Conv_2_aggregated_features[7,5]
   + 1.360898375511169 Conv_2_aggregated_features[7,6]
   - 0.1619014590978622 Conv_2_aggregated_features[7,7]
   + 0.0715495049953461 Conv_2_aggregated_features[7,8]
   + 0.1082091927528381 Conv_2_aggregated_features[7,9]
   + 1.219732761383057 Conv_2_aggregated_features[7,10]
   + 0.0949473306536674 Conv_2_aggregated_features[7,11]
   - 0.7561001181602478 Conv_2_aggregated_features[7,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[7,13]
   + 0.3085470497608185 Conv_2_aggregated_features[7,14]
   - 0.1963151693344116 Conv_2_aggregated_features[7,15] + Conv_2_t[7,4]
   = -0.681087851524353
 Conv_2_output_constraint[7,5]: 0.4891856014728546 Conv_1_Relu_ts[7,0]
   - 1.764690637588501 Conv_1_Relu_ts[7,1]
   + 1.572006106376648 Conv_1_Relu_ts[7,2]
   - 0.2603900134563446 Conv_1_Relu_ts[7,3]
   - 2.532854557037354 Conv_1_Relu_ts[7,4]
   - 0.0787733495235443 Conv_1_Relu_ts[7,5]
   + 0.3314015865325928 Conv_1_Relu_ts[7,6]
   + 0.3532952666282654 Conv_1_Relu_ts[7,7]
   + 1.495306849479675 Conv_1_Relu_ts[7,8]
   - 0.3582539260387421 Conv_1_Relu_ts[7,9]
   + 1.239243626594543 Conv_1_Relu_ts[7,10]
   + 0.0796256512403488 Conv_1_Relu_ts[7,11]
   + 1.406325340270996 Conv_1_Relu_ts[7,12]
   + 0.2874834537506104 Conv_1_Relu_ts[7,13]
   - 0.4502094686031342 Conv_1_Relu_ts[7,14]
   - 0.1813227832317352 Conv_1_Relu_ts[7,15]
   - 1.002788901329041 Conv_2_aggregated_features[7,0]
   - 1.268097281455994 Conv_2_aggregated_features[7,1]
   - 1.182242274284363 Conv_2_aggregated_features[7,2]
   - 0.3146973550319672 Conv_2_aggregated_features[7,3]
   + 1.659744143486023 Conv_2_aggregated_features[7,4]
   - 0.0238673985004425 Conv_2_aggregated_features[7,5]
   - 0.0993486419320107 Conv_2_aggregated_features[7,6]
   + 0.1021591797471046 Conv_2_aggregated_features[7,7]
   - 0.9812391996383667 Conv_2_aggregated_features[7,8]
   + 0.6361307501792908 Conv_2_aggregated_features[7,9]
   - 0.2150242179632187 Conv_2_aggregated_features[7,10]
   - 2.391498804092407 Conv_2_aggregated_features[7,11]
   + 0.9408105611801147 Conv_2_aggregated_features[7,12]
   - 0.0835467427968979 Conv_2_aggregated_features[7,13]
   - 0.7100914716720581 Conv_2_aggregated_features[7,14]
   + 0.081059992313385 Conv_2_aggregated_features[7,15] + Conv_2_t[7,5]
   = -1.552430987358093
 Conv_2_output_constraint[7,6]: 0.3276324272155762 Conv_1_Relu_ts[7,0]
   - 1.526854515075684 Conv_1_Relu_ts[7,1]
   - 0.2327080518007278 Conv_1_Relu_ts[7,2]
   + 0.2350454330444336 Conv_1_Relu_ts[7,3]
   - 2.336396932601929 Conv_1_Relu_ts[7,4]
   + 0.0498765707015991 Conv_1_Relu_ts[7,5]
   + 0.1361224502325058 Conv_1_Relu_ts[7,6]
   - 0.7505534887313843 Conv_1_Relu_ts[7,7]
   - 0.1081981733441353 Conv_1_Relu_ts[7,8]
   + 0.36195108294487 Conv_1_Relu_ts[7,9]
   + 0.266346663236618 Conv_1_Relu_ts[7,10]
   - 0.2529528141021729 Conv_1_Relu_ts[7,11]
   + 0.3400895595550537 Conv_1_Relu_ts[7,12]
   - 0.1297148019075394 Conv_1_Relu_ts[7,13]
   + 0.2151342928409576 Conv_1_Relu_ts[7,14]
   - 0.0570834763348103 Conv_1_Relu_ts[7,15]
   - 0.2712872326374054 Conv_2_aggregated_features[7,0]
   - 0.6653857827186584 Conv_2_aggregated_features[7,1]
   + 0.0704844743013382 Conv_2_aggregated_features[7,2]
   - 0.00456566177308559 Conv_2_aggregated_features[7,3]
   + 0.0726276636123657 Conv_2_aggregated_features[7,4]
   - 0.1622210741043091 Conv_2_aggregated_features[7,5]
   - 0.6965416669845581 Conv_2_aggregated_features[7,6]
   - 0.7083758115768433 Conv_2_aggregated_features[7,7]
   - 0.6968990564346313 Conv_2_aggregated_features[7,8]
   + 0.575663149356842 Conv_2_aggregated_features[7,9]
   - 0.3281366527080536 Conv_2_aggregated_features[7,10]
   - 1.37471342086792 Conv_2_aggregated_features[7,11]
   + 1.073476433753967 Conv_2_aggregated_features[7,12]
   + 0.0794747099280357 Conv_2_aggregated_features[7,13]
   + 0.5443784594535828 Conv_2_aggregated_features[7,14]
   - 0.0632283836603165 Conv_2_aggregated_features[7,15] + Conv_2_t[7,6]
   = 0.4907519221305847
 Conv_2_output_constraint[7,7]: - 0.0696223676204681 Conv_1_Relu_ts[7,0]
   - 0.1037913709878922 Conv_1_Relu_ts[7,1]
   + 0.5115458965301514 Conv_1_Relu_ts[7,2]
   - 0.1024313941597939 Conv_1_Relu_ts[7,3]
   + 0.2925211489200592 Conv_1_Relu_ts[7,4]
   + 0.1706541776657104 Conv_1_Relu_ts[7,5]
   + 0.1667102426290512 Conv_1_Relu_ts[7,6]
   + 0.2208153456449509 Conv_1_Relu_ts[7,7]
   + 0.0217245072126389 Conv_1_Relu_ts[7,8]
   + 0.2856383323669434 Conv_1_Relu_ts[7,9]
   + 0.1799796521663666 Conv_1_Relu_ts[7,10]
   + 0.2511521875858307 Conv_1_Relu_ts[7,11]
   + 0.1590938121080399 Conv_1_Relu_ts[7,12]
   - 0.1085780262947083 Conv_1_Relu_ts[7,13]
   - 0.1072387397289276 Conv_1_Relu_ts[7,14]
   + 0.0257974341511726 Conv_1_Relu_ts[7,15]
   - 0.00193481589667499 Conv_2_aggregated_features[7,0]
   - 0.0832197070121765 Conv_2_aggregated_features[7,1]
   + 0.2962732911109924 Conv_2_aggregated_features[7,2]
   + 0.1245143488049507 Conv_2_aggregated_features[7,3]
   + 0.0374857746064663 Conv_2_aggregated_features[7,4]
   - 0.2035515308380127 Conv_2_aggregated_features[7,5]
   + 0.0183010119944811 Conv_2_aggregated_features[7,6]
   - 0.0808482989668846 Conv_2_aggregated_features[7,7]
   + 0.0491516962647438 Conv_2_aggregated_features[7,8]
   - 0.1665307730436325 Conv_2_aggregated_features[7,9]
   + 0.180585503578186 Conv_2_aggregated_features[7,10]
   + 0.0900064706802368 Conv_2_aggregated_features[7,11]
   + 0.1489291191101074 Conv_2_aggregated_features[7,12]
   - 0.1824352592229843 Conv_2_aggregated_features[7,13]
   - 0.0335522592067719 Conv_2_aggregated_features[7,14]
   + 0.1226971745491028 Conv_2_aggregated_features[7,15] + Conv_2_t[7,7]
   = -0.0930573865771294
 Conv_2_output_constraint[8,0]: - 1.283687114715576 Conv_1_Relu_ts[8,0]
   - 0.7330741286277771 Conv_1_Relu_ts[8,1]
   - 2.277222394943237 Conv_1_Relu_ts[8,2]
   + 0.2110073268413544 Conv_1_Relu_ts[8,3]
   - 2.244216442108154 Conv_1_Relu_ts[8,4]
   + 0.1934826672077179 Conv_1_Relu_ts[8,5]
   - 1.220147132873535 Conv_1_Relu_ts[8,6]
   - 0.2545500099658966 Conv_1_Relu_ts[8,7]
   + 1.472832441329956 Conv_1_Relu_ts[8,8]
   - 1.084863781929016 Conv_1_Relu_ts[8,9]
   + 3.159408807754517 Conv_1_Relu_ts[8,10]
   - 1.101905822753906 Conv_1_Relu_ts[8,11]
   + 0.4563636183738708 Conv_1_Relu_ts[8,12]
   - 0.1066205129027367 Conv_1_Relu_ts[8,13]
   + 0.1282547265291214 Conv_1_Relu_ts[8,14]
   - 0.1789750158786774 Conv_1_Relu_ts[8,15]
   + 1.150216221809387 Conv_2_aggregated_features[8,0]
   + 0.7049534916877747 Conv_2_aggregated_features[8,1]
   - 0.6801650524139404 Conv_2_aggregated_features[8,2]
   + 0.0596005544066429 Conv_2_aggregated_features[8,3]
   + 1.205108404159546 Conv_2_aggregated_features[8,4]
   - 0.0892623364925385 Conv_2_aggregated_features[8,5]
   + 1.500906825065613 Conv_2_aggregated_features[8,6]
   - 0.4810487031936646 Conv_2_aggregated_features[8,7]
   + 0.3540807068347931 Conv_2_aggregated_features[8,8]
   - 0.3831521570682526 Conv_2_aggregated_features[8,9]
   - 0.5447250604629517 Conv_2_aggregated_features[8,10]
   + 2.773675441741943 Conv_2_aggregated_features[8,11]
   + 0.4864743649959564 Conv_2_aggregated_features[8,12]
   + 0.2153479158878326 Conv_2_aggregated_features[8,13]
   + 1.067057013511658 Conv_2_aggregated_features[8,14]
   - 0.2962089478969574 Conv_2_aggregated_features[8,15] + Conv_2_t[8,0]
   = -0.1796741187572479
 Conv_2_output_constraint[8,1]: - 0.8503658175468445 Conv_1_Relu_ts[8,0]
   + 0.1598657667636871 Conv_1_Relu_ts[8,1]
   + 0.740400493144989 Conv_1_Relu_ts[8,2]
   - 0.0143840508535504 Conv_1_Relu_ts[8,3]
   + 0.418368011713028 Conv_1_Relu_ts[8,4]
   - 0.0816935896873474 Conv_1_Relu_ts[8,5]
   - 0.6345715522766113 Conv_1_Relu_ts[8,6]
   - 0.1658433526754379 Conv_1_Relu_ts[8,7]
   + 0.2588719427585602 Conv_1_Relu_ts[8,8]
   - 0.3725545704364777 Conv_1_Relu_ts[8,9]
   - 0.1093620359897614 Conv_1_Relu_ts[8,10]
   - 0.229639545083046 Conv_1_Relu_ts[8,11]
   + 2.353754758834839 Conv_1_Relu_ts[8,12]
   + 0.275741845369339 Conv_1_Relu_ts[8,13]
   + 0.2292672693729401 Conv_1_Relu_ts[8,14]
   - 0.085102416574955 Conv_1_Relu_ts[8,15]
   + 1.31701123714447 Conv_2_aggregated_features[8,0]
   + 0.5385994911193848 Conv_2_aggregated_features[8,1]
   - 0.8459239602088928 Conv_2_aggregated_features[8,2]
   + 0.1865081340074539 Conv_2_aggregated_features[8,3]
   + 0.8385969400405884 Conv_2_aggregated_features[8,4]
   - 0.1273856163024902 Conv_2_aggregated_features[8,5]
   + 2.53032374382019 Conv_2_aggregated_features[8,6]
   - 0.3543893694877625 Conv_2_aggregated_features[8,7]
   + 0.1923532038927078 Conv_2_aggregated_features[8,8]
   + 1.131453990936279 Conv_2_aggregated_features[8,9]
   - 0.3651714324951172 Conv_2_aggregated_features[8,10]
   + 1.542536735534668 Conv_2_aggregated_features[8,11]
   - 0.4610330760478973 Conv_2_aggregated_features[8,12]
   - 0.00123762537259609 Conv_2_aggregated_features[8,13]
   + 0.375784307718277 Conv_2_aggregated_features[8,14]
   + 0.0822948440909386 Conv_2_aggregated_features[8,15] + Conv_2_t[8,1]
   = -0.280759334564209
 Conv_2_output_constraint[8,2]: 1.746561527252197 Conv_1_Relu_ts[8,0]
   + 0.814217746257782 Conv_1_Relu_ts[8,1]
   - 1.465980052947998 Conv_1_Relu_ts[8,2]
   - 0.2278070449829102 Conv_1_Relu_ts[8,3]
   + 1.037705659866333 Conv_1_Relu_ts[8,4]
   - 0.2538289427757263 Conv_1_Relu_ts[8,5]
   + 0.6437492966651917 Conv_1_Relu_ts[8,6]
   - 0.4347417056560516 Conv_1_Relu_ts[8,7]
   + 0.2060042917728424 Conv_1_Relu_ts[8,8]
   + 0.8382987976074219 Conv_1_Relu_ts[8,9]
   - 0.6877544522285461 Conv_1_Relu_ts[8,10]
   + 1.350347280502319 Conv_1_Relu_ts[8,11]
   + 2.458039999008179 Conv_1_Relu_ts[8,12]
   - 0.1695821583271027 Conv_1_Relu_ts[8,13]
   + 0.2783661186695099 Conv_1_Relu_ts[8,14]
   - 0.1666328907012939 Conv_1_Relu_ts[8,15]
   - 0.4272331893444061 Conv_2_aggregated_features[8,0]
   + 0.8438286781311035 Conv_2_aggregated_features[8,1]
   - 1.847808957099915 Conv_2_aggregated_features[8,2]
   - 0.1186513304710388 Conv_2_aggregated_features[8,3]
   + 0.6336349844932556 Conv_2_aggregated_features[8,4]
   + 0.0602188408374786 Conv_2_aggregated_features[8,5]
   - 0.2530380189418793 Conv_2_aggregated_features[8,6]
   - 0.5760374069213867 Conv_2_aggregated_features[8,7]
   - 0.8055176734924316 Conv_2_aggregated_features[8,8]
   + 0.4366684854030609 Conv_2_aggregated_features[8,9]
   - 1.177173972129822 Conv_2_aggregated_features[8,10]
   - 0.4784670472145081 Conv_2_aggregated_features[8,11]
   + 2.325191736221313 Conv_2_aggregated_features[8,12]
   - 0.2128628194332123 Conv_2_aggregated_features[8,13]
   - 0.1453570574522018 Conv_2_aggregated_features[8,14]
   - 0.118608683347702 Conv_2_aggregated_features[8,15] + Conv_2_t[8,2]
   = 1.507851004600525
 Conv_2_output_constraint[8,3]: - 0.527165412902832 Conv_1_Relu_ts[8,0]
   - 1.389649629592896 Conv_1_Relu_ts[8,1]
   - 0.6640920639038086 Conv_1_Relu_ts[8,2]
   - 0.0634000822901726 Conv_1_Relu_ts[8,3]
   - 0.4163890480995178 Conv_1_Relu_ts[8,4]
   + 0.0613848567008972 Conv_1_Relu_ts[8,5]
   + 1.790452718734741 Conv_1_Relu_ts[8,6]
   - 0.1128331422805786 Conv_1_Relu_ts[8,7]
   + 2.12717342376709 Conv_1_Relu_ts[8,8]
   - 0.1405593752861023 Conv_1_Relu_ts[8,9]
   - 0.1852237284183502 Conv_1_Relu_ts[8,10]
   - 0.2454802542924881 Conv_1_Relu_ts[8,11]
   - 1.97015655040741 Conv_1_Relu_ts[8,12]
   + 0.3291732370853424 Conv_1_Relu_ts[8,13]
   - 0.5780640244483948 Conv_1_Relu_ts[8,14]
   - 0.1067387536168098 Conv_1_Relu_ts[8,15]
   - 0.1432812511920929 Conv_2_aggregated_features[8,0]
   - 0.4248973727226257 Conv_2_aggregated_features[8,1]
   + 1.125378131866455 Conv_2_aggregated_features[8,2]
   + 0.1260246187448502 Conv_2_aggregated_features[8,3]
   - 0.5538251996040344 Conv_2_aggregated_features[8,4]
   + 0.1312114000320435 Conv_2_aggregated_features[8,5]
   - 0.9671168327331543 Conv_2_aggregated_features[8,6]
   - 0.0926898792386055 Conv_2_aggregated_features[8,7]
   - 0.1030022576451302 Conv_2_aggregated_features[8,8]
   - 1.69473397731781 Conv_2_aggregated_features[8,9]
   + 2.313785552978516 Conv_2_aggregated_features[8,10]
   - 0.3551321625709534 Conv_2_aggregated_features[8,11]
   - 0.441056102514267 Conv_2_aggregated_features[8,12]
   - 0.0765185728669167 Conv_2_aggregated_features[8,13]
   - 0.5983367562294006 Conv_2_aggregated_features[8,14]
   + 0.023831807076931 Conv_2_aggregated_features[8,15] + Conv_2_t[8,3]
   = -0.072962760925293
 Conv_2_output_constraint[8,4]: 0.5219888687133789 Conv_1_Relu_ts[8,0]
   - 1.025602698326111 Conv_1_Relu_ts[8,1]
   - 0.4347750842571259 Conv_1_Relu_ts[8,2]
   + 0.2773195803165436 Conv_1_Relu_ts[8,3]
   - 2.5947265625 Conv_1_Relu_ts[8,4]
   - 0.2330431044101715 Conv_1_Relu_ts[8,5]
   + 0.4634701013565063 Conv_1_Relu_ts[8,6]
   - 0.2591606974601746 Conv_1_Relu_ts[8,7]
   + 0.715503990650177 Conv_1_Relu_ts[8,8]
   - 0.1986619830131531 Conv_1_Relu_ts[8,9]
   - 0.9099530577659607 Conv_1_Relu_ts[8,10]
   - 0.2335229516029358 Conv_1_Relu_ts[8,11]
   - 0.2992779910564423 Conv_1_Relu_ts[8,12]
   + 0.000100307 Conv_1_Relu_ts[8,13]
   - 1.00616979598999 Conv_1_Relu_ts[8,14]
   + 0.2120778858661652 Conv_1_Relu_ts[8,15]
   - 0.0600817315280437 Conv_2_aggregated_features[8,0]
   - 0.099200963973999 Conv_2_aggregated_features[8,1]
   - 0.4625514149665833 Conv_2_aggregated_features[8,2]
   - 0.1394201517105103 Conv_2_aggregated_features[8,3]
   + 1.153588056564331 Conv_2_aggregated_features[8,4]
   - 0.0127062797546387 Conv_2_aggregated_features[8,5]
   + 1.360898375511169 Conv_2_aggregated_features[8,6]
   - 0.1619014590978622 Conv_2_aggregated_features[8,7]
   + 0.0715495049953461 Conv_2_aggregated_features[8,8]
   + 0.1082091927528381 Conv_2_aggregated_features[8,9]
   + 1.219732761383057 Conv_2_aggregated_features[8,10]
   + 0.0949473306536674 Conv_2_aggregated_features[8,11]
   - 0.7561001181602478 Conv_2_aggregated_features[8,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[8,13]
   + 0.3085470497608185 Conv_2_aggregated_features[8,14]
   - 0.1963151693344116 Conv_2_aggregated_features[8,15] + Conv_2_t[8,4]
   = -0.681087851524353
 Conv_2_output_constraint[8,5]: 0.4891856014728546 Conv_1_Relu_ts[8,0]
   - 1.764690637588501 Conv_1_Relu_ts[8,1]
   + 1.572006106376648 Conv_1_Relu_ts[8,2]
   - 0.2603900134563446 Conv_1_Relu_ts[8,3]
   - 2.532854557037354 Conv_1_Relu_ts[8,4]
   - 0.0787733495235443 Conv_1_Relu_ts[8,5]
   + 0.3314015865325928 Conv_1_Relu_ts[8,6]
   + 0.3532952666282654 Conv_1_Relu_ts[8,7]
   + 1.495306849479675 Conv_1_Relu_ts[8,8]
   - 0.3582539260387421 Conv_1_Relu_ts[8,9]
   + 1.239243626594543 Conv_1_Relu_ts[8,10]
   + 0.0796256512403488 Conv_1_Relu_ts[8,11]
   + 1.406325340270996 Conv_1_Relu_ts[8,12]
   + 0.2874834537506104 Conv_1_Relu_ts[8,13]
   - 0.4502094686031342 Conv_1_Relu_ts[8,14]
   - 0.1813227832317352 Conv_1_Relu_ts[8,15]
   - 1.002788901329041 Conv_2_aggregated_features[8,0]
   - 1.268097281455994 Conv_2_aggregated_features[8,1]
   - 1.182242274284363 Conv_2_aggregated_features[8,2]
   - 0.3146973550319672 Conv_2_aggregated_features[8,3]
   + 1.659744143486023 Conv_2_aggregated_features[8,4]
   - 0.0238673985004425 Conv_2_aggregated_features[8,5]
   - 0.0993486419320107 Conv_2_aggregated_features[8,6]
   + 0.1021591797471046 Conv_2_aggregated_features[8,7]
   - 0.9812391996383667 Conv_2_aggregated_features[8,8]
   + 0.6361307501792908 Conv_2_aggregated_features[8,9]
   - 0.2150242179632187 Conv_2_aggregated_features[8,10]
   - 2.391498804092407 Conv_2_aggregated_features[8,11]
   + 0.9408105611801147 Conv_2_aggregated_features[8,12]
   - 0.0835467427968979 Conv_2_aggregated_features[8,13]
   - 0.7100914716720581 Conv_2_aggregated_features[8,14]
   + 0.081059992313385 Conv_2_aggregated_features[8,15] + Conv_2_t[8,5]
   = -1.552430987358093
 Conv_2_output_constraint[8,6]: 0.3276324272155762 Conv_1_Relu_ts[8,0]
   - 1.526854515075684 Conv_1_Relu_ts[8,1]
   - 0.2327080518007278 Conv_1_Relu_ts[8,2]
   + 0.2350454330444336 Conv_1_Relu_ts[8,3]
   - 2.336396932601929 Conv_1_Relu_ts[8,4]
   + 0.0498765707015991 Conv_1_Relu_ts[8,5]
   + 0.1361224502325058 Conv_1_Relu_ts[8,6]
   - 0.7505534887313843 Conv_1_Relu_ts[8,7]
   - 0.1081981733441353 Conv_1_Relu_ts[8,8]
   + 0.36195108294487 Conv_1_Relu_ts[8,9]
   + 0.266346663236618 Conv_1_Relu_ts[8,10]
   - 0.2529528141021729 Conv_1_Relu_ts[8,11]
   + 0.3400895595550537 Conv_1_Relu_ts[8,12]
   - 0.1297148019075394 Conv_1_Relu_ts[8,13]
   + 0.2151342928409576 Conv_1_Relu_ts[8,14]
   - 0.0570834763348103 Conv_1_Relu_ts[8,15]
   - 0.2712872326374054 Conv_2_aggregated_features[8,0]
   - 0.6653857827186584 Conv_2_aggregated_features[8,1]
   + 0.0704844743013382 Conv_2_aggregated_features[8,2]
   - 0.00456566177308559 Conv_2_aggregated_features[8,3]
   + 0.0726276636123657 Conv_2_aggregated_features[8,4]
   - 0.1622210741043091 Conv_2_aggregated_features[8,5]
   - 0.6965416669845581 Conv_2_aggregated_features[8,6]
   - 0.7083758115768433 Conv_2_aggregated_features[8,7]
   - 0.6968990564346313 Conv_2_aggregated_features[8,8]
   + 0.575663149356842 Conv_2_aggregated_features[8,9]
   - 0.3281366527080536 Conv_2_aggregated_features[8,10]
   - 1.37471342086792 Conv_2_aggregated_features[8,11]
   + 1.073476433753967 Conv_2_aggregated_features[8,12]
   + 0.0794747099280357 Conv_2_aggregated_features[8,13]
   + 0.5443784594535828 Conv_2_aggregated_features[8,14]
   - 0.0632283836603165 Conv_2_aggregated_features[8,15] + Conv_2_t[8,6]
   = 0.4907519221305847
 Conv_2_output_constraint[8,7]: - 0.0696223676204681 Conv_1_Relu_ts[8,0]
   - 0.1037913709878922 Conv_1_Relu_ts[8,1]
   + 0.5115458965301514 Conv_1_Relu_ts[8,2]
   - 0.1024313941597939 Conv_1_Relu_ts[8,3]
   + 0.2925211489200592 Conv_1_Relu_ts[8,4]
   + 0.1706541776657104 Conv_1_Relu_ts[8,5]
   + 0.1667102426290512 Conv_1_Relu_ts[8,6]
   + 0.2208153456449509 Conv_1_Relu_ts[8,7]
   + 0.0217245072126389 Conv_1_Relu_ts[8,8]
   + 0.2856383323669434 Conv_1_Relu_ts[8,9]
   + 0.1799796521663666 Conv_1_Relu_ts[8,10]
   + 0.2511521875858307 Conv_1_Relu_ts[8,11]
   + 0.1590938121080399 Conv_1_Relu_ts[8,12]
   - 0.1085780262947083 Conv_1_Relu_ts[8,13]
   - 0.1072387397289276 Conv_1_Relu_ts[8,14]
   + 0.0257974341511726 Conv_1_Relu_ts[8,15]
   - 0.00193481589667499 Conv_2_aggregated_features[8,0]
   - 0.0832197070121765 Conv_2_aggregated_features[8,1]
   + 0.2962732911109924 Conv_2_aggregated_features[8,2]
   + 0.1245143488049507 Conv_2_aggregated_features[8,3]
   + 0.0374857746064663 Conv_2_aggregated_features[8,4]
   - 0.2035515308380127 Conv_2_aggregated_features[8,5]
   + 0.0183010119944811 Conv_2_aggregated_features[8,6]
   - 0.0808482989668846 Conv_2_aggregated_features[8,7]
   + 0.0491516962647438 Conv_2_aggregated_features[8,8]
   - 0.1665307730436325 Conv_2_aggregated_features[8,9]
   + 0.180585503578186 Conv_2_aggregated_features[8,10]
   + 0.0900064706802368 Conv_2_aggregated_features[8,11]
   + 0.1489291191101074 Conv_2_aggregated_features[8,12]
   - 0.1824352592229843 Conv_2_aggregated_features[8,13]
   - 0.0335522592067719 Conv_2_aggregated_features[8,14]
   + 0.1226971745491028 Conv_2_aggregated_features[8,15] + Conv_2_t[8,7]
   = -0.0930573865771294
 Conv_2_output_constraint[9,0]: - 1.283687114715576 Conv_1_Relu_ts[9,0]
   - 0.7330741286277771 Conv_1_Relu_ts[9,1]
   - 2.277222394943237 Conv_1_Relu_ts[9,2]
   + 0.2110073268413544 Conv_1_Relu_ts[9,3]
   - 2.244216442108154 Conv_1_Relu_ts[9,4]
   + 0.1934826672077179 Conv_1_Relu_ts[9,5]
   - 1.220147132873535 Conv_1_Relu_ts[9,6]
   - 0.2545500099658966 Conv_1_Relu_ts[9,7]
   + 1.472832441329956 Conv_1_Relu_ts[9,8]
   - 1.084863781929016 Conv_1_Relu_ts[9,9]
   + 3.159408807754517 Conv_1_Relu_ts[9,10]
   - 1.101905822753906 Conv_1_Relu_ts[9,11]
   + 0.4563636183738708 Conv_1_Relu_ts[9,12]
   - 0.1066205129027367 Conv_1_Relu_ts[9,13]
   + 0.1282547265291214 Conv_1_Relu_ts[9,14]
   - 0.1789750158786774 Conv_1_Relu_ts[9,15]
   + 1.150216221809387 Conv_2_aggregated_features[9,0]
   + 0.7049534916877747 Conv_2_aggregated_features[9,1]
   - 0.6801650524139404 Conv_2_aggregated_features[9,2]
   + 0.0596005544066429 Conv_2_aggregated_features[9,3]
   + 1.205108404159546 Conv_2_aggregated_features[9,4]
   - 0.0892623364925385 Conv_2_aggregated_features[9,5]
   + 1.500906825065613 Conv_2_aggregated_features[9,6]
   - 0.4810487031936646 Conv_2_aggregated_features[9,7]
   + 0.3540807068347931 Conv_2_aggregated_features[9,8]
   - 0.3831521570682526 Conv_2_aggregated_features[9,9]
   - 0.5447250604629517 Conv_2_aggregated_features[9,10]
   + 2.773675441741943 Conv_2_aggregated_features[9,11]
   + 0.4864743649959564 Conv_2_aggregated_features[9,12]
   + 0.2153479158878326 Conv_2_aggregated_features[9,13]
   + 1.067057013511658 Conv_2_aggregated_features[9,14]
   - 0.2962089478969574 Conv_2_aggregated_features[9,15] + Conv_2_t[9,0]
   = -0.1796741187572479
 Conv_2_output_constraint[9,1]: - 0.8503658175468445 Conv_1_Relu_ts[9,0]
   + 0.1598657667636871 Conv_1_Relu_ts[9,1]
   + 0.740400493144989 Conv_1_Relu_ts[9,2]
   - 0.0143840508535504 Conv_1_Relu_ts[9,3]
   + 0.418368011713028 Conv_1_Relu_ts[9,4]
   - 0.0816935896873474 Conv_1_Relu_ts[9,5]
   - 0.6345715522766113 Conv_1_Relu_ts[9,6]
   - 0.1658433526754379 Conv_1_Relu_ts[9,7]
   + 0.2588719427585602 Conv_1_Relu_ts[9,8]
   - 0.3725545704364777 Conv_1_Relu_ts[9,9]
   - 0.1093620359897614 Conv_1_Relu_ts[9,10]
   - 0.229639545083046 Conv_1_Relu_ts[9,11]
   + 2.353754758834839 Conv_1_Relu_ts[9,12]
   + 0.275741845369339 Conv_1_Relu_ts[9,13]
   + 0.2292672693729401 Conv_1_Relu_ts[9,14]
   - 0.085102416574955 Conv_1_Relu_ts[9,15]
   + 1.31701123714447 Conv_2_aggregated_features[9,0]
   + 0.5385994911193848 Conv_2_aggregated_features[9,1]
   - 0.8459239602088928 Conv_2_aggregated_features[9,2]
   + 0.1865081340074539 Conv_2_aggregated_features[9,3]
   + 0.8385969400405884 Conv_2_aggregated_features[9,4]
   - 0.1273856163024902 Conv_2_aggregated_features[9,5]
   + 2.53032374382019 Conv_2_aggregated_features[9,6]
   - 0.3543893694877625 Conv_2_aggregated_features[9,7]
   + 0.1923532038927078 Conv_2_aggregated_features[9,8]
   + 1.131453990936279 Conv_2_aggregated_features[9,9]
   - 0.3651714324951172 Conv_2_aggregated_features[9,10]
   + 1.542536735534668 Conv_2_aggregated_features[9,11]
   - 0.4610330760478973 Conv_2_aggregated_features[9,12]
   - 0.00123762537259609 Conv_2_aggregated_features[9,13]
   + 0.375784307718277 Conv_2_aggregated_features[9,14]
   + 0.0822948440909386 Conv_2_aggregated_features[9,15] + Conv_2_t[9,1]
   = -0.280759334564209
 Conv_2_output_constraint[9,2]: 1.746561527252197 Conv_1_Relu_ts[9,0]
   + 0.814217746257782 Conv_1_Relu_ts[9,1]
   - 1.465980052947998 Conv_1_Relu_ts[9,2]
   - 0.2278070449829102 Conv_1_Relu_ts[9,3]
   + 1.037705659866333 Conv_1_Relu_ts[9,4]
   - 0.2538289427757263 Conv_1_Relu_ts[9,5]
   + 0.6437492966651917 Conv_1_Relu_ts[9,6]
   - 0.4347417056560516 Conv_1_Relu_ts[9,7]
   + 0.2060042917728424 Conv_1_Relu_ts[9,8]
   + 0.8382987976074219 Conv_1_Relu_ts[9,9]
   - 0.6877544522285461 Conv_1_Relu_ts[9,10]
   + 1.350347280502319 Conv_1_Relu_ts[9,11]
   + 2.458039999008179 Conv_1_Relu_ts[9,12]
   - 0.1695821583271027 Conv_1_Relu_ts[9,13]
   + 0.2783661186695099 Conv_1_Relu_ts[9,14]
   - 0.1666328907012939 Conv_1_Relu_ts[9,15]
   - 0.4272331893444061 Conv_2_aggregated_features[9,0]
   + 0.8438286781311035 Conv_2_aggregated_features[9,1]
   - 1.847808957099915 Conv_2_aggregated_features[9,2]
   - 0.1186513304710388 Conv_2_aggregated_features[9,3]
   + 0.6336349844932556 Conv_2_aggregated_features[9,4]
   + 0.0602188408374786 Conv_2_aggregated_features[9,5]
   - 0.2530380189418793 Conv_2_aggregated_features[9,6]
   - 0.5760374069213867 Conv_2_aggregated_features[9,7]
   - 0.8055176734924316 Conv_2_aggregated_features[9,8]
   + 0.4366684854030609 Conv_2_aggregated_features[9,9]
   - 1.177173972129822 Conv_2_aggregated_features[9,10]
   - 0.4784670472145081 Conv_2_aggregated_features[9,11]
   + 2.325191736221313 Conv_2_aggregated_features[9,12]
   - 0.2128628194332123 Conv_2_aggregated_features[9,13]
   - 0.1453570574522018 Conv_2_aggregated_features[9,14]
   - 0.118608683347702 Conv_2_aggregated_features[9,15] + Conv_2_t[9,2]
   = 1.507851004600525
 Conv_2_output_constraint[9,3]: - 0.527165412902832 Conv_1_Relu_ts[9,0]
   - 1.389649629592896 Conv_1_Relu_ts[9,1]
   - 0.6640920639038086 Conv_1_Relu_ts[9,2]
   - 0.0634000822901726 Conv_1_Relu_ts[9,3]
   - 0.4163890480995178 Conv_1_Relu_ts[9,4]
   + 0.0613848567008972 Conv_1_Relu_ts[9,5]
   + 1.790452718734741 Conv_1_Relu_ts[9,6]
   - 0.1128331422805786 Conv_1_Relu_ts[9,7]
   + 2.12717342376709 Conv_1_Relu_ts[9,8]
   - 0.1405593752861023 Conv_1_Relu_ts[9,9]
   - 0.1852237284183502 Conv_1_Relu_ts[9,10]
   - 0.2454802542924881 Conv_1_Relu_ts[9,11]
   - 1.97015655040741 Conv_1_Relu_ts[9,12]
   + 0.3291732370853424 Conv_1_Relu_ts[9,13]
   - 0.5780640244483948 Conv_1_Relu_ts[9,14]
   - 0.1067387536168098 Conv_1_Relu_ts[9,15]
   - 0.1432812511920929 Conv_2_aggregated_features[9,0]
   - 0.4248973727226257 Conv_2_aggregated_features[9,1]
   + 1.125378131866455 Conv_2_aggregated_features[9,2]
   + 0.1260246187448502 Conv_2_aggregated_features[9,3]
   - 0.5538251996040344 Conv_2_aggregated_features[9,4]
   + 0.1312114000320435 Conv_2_aggregated_features[9,5]
   - 0.9671168327331543 Conv_2_aggregated_features[9,6]
   - 0.0926898792386055 Conv_2_aggregated_features[9,7]
   - 0.1030022576451302 Conv_2_aggregated_features[9,8]
   - 1.69473397731781 Conv_2_aggregated_features[9,9]
   + 2.313785552978516 Conv_2_aggregated_features[9,10]
   - 0.3551321625709534 Conv_2_aggregated_features[9,11]
   - 0.441056102514267 Conv_2_aggregated_features[9,12]
   - 0.0765185728669167 Conv_2_aggregated_features[9,13]
   - 0.5983367562294006 Conv_2_aggregated_features[9,14]
   + 0.023831807076931 Conv_2_aggregated_features[9,15] + Conv_2_t[9,3]
   = -0.072962760925293
 Conv_2_output_constraint[9,4]: 0.5219888687133789 Conv_1_Relu_ts[9,0]
   - 1.025602698326111 Conv_1_Relu_ts[9,1]
   - 0.4347750842571259 Conv_1_Relu_ts[9,2]
   + 0.2773195803165436 Conv_1_Relu_ts[9,3]
   - 2.5947265625 Conv_1_Relu_ts[9,4]
   - 0.2330431044101715 Conv_1_Relu_ts[9,5]
   + 0.4634701013565063 Conv_1_Relu_ts[9,6]
   - 0.2591606974601746 Conv_1_Relu_ts[9,7]
   + 0.715503990650177 Conv_1_Relu_ts[9,8]
   - 0.1986619830131531 Conv_1_Relu_ts[9,9]
   - 0.9099530577659607 Conv_1_Relu_ts[9,10]
   - 0.2335229516029358 Conv_1_Relu_ts[9,11]
   - 0.2992779910564423 Conv_1_Relu_ts[9,12]
   + 0.000100307 Conv_1_Relu_ts[9,13]
   - 1.00616979598999 Conv_1_Relu_ts[9,14]
   + 0.2120778858661652 Conv_1_Relu_ts[9,15]
   - 0.0600817315280437 Conv_2_aggregated_features[9,0]
   - 0.099200963973999 Conv_2_aggregated_features[9,1]
   - 0.4625514149665833 Conv_2_aggregated_features[9,2]
   - 0.1394201517105103 Conv_2_aggregated_features[9,3]
   + 1.153588056564331 Conv_2_aggregated_features[9,4]
   - 0.0127062797546387 Conv_2_aggregated_features[9,5]
   + 1.360898375511169 Conv_2_aggregated_features[9,6]
   - 0.1619014590978622 Conv_2_aggregated_features[9,7]
   + 0.0715495049953461 Conv_2_aggregated_features[9,8]
   + 0.1082091927528381 Conv_2_aggregated_features[9,9]
   + 1.219732761383057 Conv_2_aggregated_features[9,10]
   + 0.0949473306536674 Conv_2_aggregated_features[9,11]
   - 0.7561001181602478 Conv_2_aggregated_features[9,12]
   - 1.0562912939349189e-04 Conv_2_aggregated_features[9,13]
   + 0.3085470497608185 Conv_2_aggregated_features[9,14]
   - 0.1963151693344116 Conv_2_aggregated_features[9,15] + Conv_2_t[9,4]
   = -0.681087851524353
 Conv_2_output_constraint[9,5]: 0.4891856014728546 Conv_1_Relu_ts[9,0]
   - 1.764690637588501 Conv_1_Relu_ts[9,1]
   + 1.572006106376648 Conv_1_Relu_ts[9,2]
   - 0.2603900134563446 Conv_1_Relu_ts[9,3]
   - 2.532854557037354 Conv_1_Relu_ts[9,4]
   - 0.0787733495235443 Conv_1_Relu_ts[9,5]
   + 0.3314015865325928 Conv_1_Relu_ts[9,6]
   + 0.3532952666282654 Conv_1_Relu_ts[9,7]
   + 1.495306849479675 Conv_1_Relu_ts[9,8]
   - 0.3582539260387421 Conv_1_Relu_ts[9,9]
   + 1.239243626594543 Conv_1_Relu_ts[9,10]
   + 0.0796256512403488 Conv_1_Relu_ts[9,11]
   + 1.406325340270996 Conv_1_Relu_ts[9,12]
   + 0.2874834537506104 Conv_1_Relu_ts[9,13]
   - 0.4502094686031342 Conv_1_Relu_ts[9,14]
   - 0.1813227832317352 Conv_1_Relu_ts[9,15]
   - 1.002788901329041 Conv_2_aggregated_features[9,0]
   - 1.268097281455994 Conv_2_aggregated_features[9,1]
   - 1.182242274284363 Conv_2_aggregated_features[9,2]
   - 0.3146973550319672 Conv_2_aggregated_features[9,3]
   + 1.659744143486023 Conv_2_aggregated_features[9,4]
   - 0.0238673985004425 Conv_2_aggregated_features[9,5]
   - 0.0993486419320107 Conv_2_aggregated_features[9,6]
   + 0.1021591797471046 Conv_2_aggregated_features[9,7]
   - 0.9812391996383667 Conv_2_aggregated_features[9,8]
   + 0.6361307501792908 Conv_2_aggregated_features[9,9]
   - 0.2150242179632187 Conv_2_aggregated_features[9,10]
   - 2.391498804092407 Conv_2_aggregated_features[9,11]
   + 0.9408105611801147 Conv_2_aggregated_features[9,12]
   - 0.0835467427968979 Conv_2_aggregated_features[9,13]
   - 0.7100914716720581 Conv_2_aggregated_features[9,14]
   + 0.081059992313385 Conv_2_aggregated_features[9,15] + Conv_2_t[9,5]
   = -1.552430987358093
 Conv_2_output_constraint[9,6]: 0.3276324272155762 Conv_1_Relu_ts[9,0]
   - 1.526854515075684 Conv_1_Relu_ts[9,1]
   - 0.2327080518007278 Conv_1_Relu_ts[9,2]
   + 0.2350454330444336 Conv_1_Relu_ts[9,3]
   - 2.336396932601929 Conv_1_Relu_ts[9,4]
   + 0.0498765707015991 Conv_1_Relu_ts[9,5]
   + 0.1361224502325058 Conv_1_Relu_ts[9,6]
   - 0.7505534887313843 Conv_1_Relu_ts[9,7]
   - 0.1081981733441353 Conv_1_Relu_ts[9,8]
   + 0.36195108294487 Conv_1_Relu_ts[9,9]
   + 0.266346663236618 Conv_1_Relu_ts[9,10]
   - 0.2529528141021729 Conv_1_Relu_ts[9,11]
   + 0.3400895595550537 Conv_1_Relu_ts[9,12]
   - 0.1297148019075394 Conv_1_Relu_ts[9,13]
   + 0.2151342928409576 Conv_1_Relu_ts[9,14]
   - 0.0570834763348103 Conv_1_Relu_ts[9,15]
   - 0.2712872326374054 Conv_2_aggregated_features[9,0]
   - 0.6653857827186584 Conv_2_aggregated_features[9,1]
   + 0.0704844743013382 Conv_2_aggregated_features[9,2]
   - 0.00456566177308559 Conv_2_aggregated_features[9,3]
   + 0.0726276636123657 Conv_2_aggregated_features[9,4]
   - 0.1622210741043091 Conv_2_aggregated_features[9,5]
   - 0.6965416669845581 Conv_2_aggregated_features[9,6]
   - 0.7083758115768433 Conv_2_aggregated_features[9,7]
   - 0.6968990564346313 Conv_2_aggregated_features[9,8]
   + 0.575663149356842 Conv_2_aggregated_features[9,9]
   - 0.3281366527080536 Conv_2_aggregated_features[9,10]
   - 1.37471342086792 Conv_2_aggregated_features[9,11]
   + 1.073476433753967 Conv_2_aggregated_features[9,12]
   + 0.0794747099280357 Conv_2_aggregated_features[9,13]
   + 0.5443784594535828 Conv_2_aggregated_features[9,14]
   - 0.0632283836603165 Conv_2_aggregated_features[9,15] + Conv_2_t[9,6]
   = 0.4907519221305847
 Conv_2_output_constraint[9,7]: - 0.0696223676204681 Conv_1_Relu_ts[9,0]
   - 0.1037913709878922 Conv_1_Relu_ts[9,1]
   + 0.5115458965301514 Conv_1_Relu_ts[9,2]
   - 0.1024313941597939 Conv_1_Relu_ts[9,3]
   + 0.2925211489200592 Conv_1_Relu_ts[9,4]
   + 0.1706541776657104 Conv_1_Relu_ts[9,5]
   + 0.1667102426290512 Conv_1_Relu_ts[9,6]
   + 0.2208153456449509 Conv_1_Relu_ts[9,7]
   + 0.0217245072126389 Conv_1_Relu_ts[9,8]
   + 0.2856383323669434 Conv_1_Relu_ts[9,9]
   + 0.1799796521663666 Conv_1_Relu_ts[9,10]
   + 0.2511521875858307 Conv_1_Relu_ts[9,11]
   + 0.1590938121080399 Conv_1_Relu_ts[9,12]
   - 0.1085780262947083 Conv_1_Relu_ts[9,13]
   - 0.1072387397289276 Conv_1_Relu_ts[9,14]
   + 0.0257974341511726 Conv_1_Relu_ts[9,15]
   - 0.00193481589667499 Conv_2_aggregated_features[9,0]
   - 0.0832197070121765 Conv_2_aggregated_features[9,1]
   + 0.2962732911109924 Conv_2_aggregated_features[9,2]
   + 0.1245143488049507 Conv_2_aggregated_features[9,3]
   + 0.0374857746064663 Conv_2_aggregated_features[9,4]
   - 0.2035515308380127 Conv_2_aggregated_features[9,5]
   + 0.0183010119944811 Conv_2_aggregated_features[9,6]
   - 0.0808482989668846 Conv_2_aggregated_features[9,7]
   + 0.0491516962647438 Conv_2_aggregated_features[9,8]
   - 0.1665307730436325 Conv_2_aggregated_features[9,9]
   + 0.180585503578186 Conv_2_aggregated_features[9,10]
   + 0.0900064706802368 Conv_2_aggregated_features[9,11]
   + 0.1489291191101074 Conv_2_aggregated_features[9,12]
   - 0.1824352592229843 Conv_2_aggregated_features[9,13]
   - 0.0335522592067719 Conv_2_aggregated_features[9,14]
   + 0.1226971745491028 Conv_2_aggregated_features[9,15] + Conv_2_t[9,7]
   = -0.0930573865771294
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
 Conv_2_Relu_constraint_1[8,0]: - Conv_2_t[8,0] + Conv_2_Relu_ts[8,0]
   - Conv_2_Relu_ss[8,0] = 0
 Conv_2_Relu_constraint_1[8,1]: - Conv_2_t[8,1] + Conv_2_Relu_ts[8,1]
   - Conv_2_Relu_ss[8,1] = 0
 Conv_2_Relu_constraint_1[8,2]: - Conv_2_t[8,2] + Conv_2_Relu_ts[8,2]
   - Conv_2_Relu_ss[8,2] = 0
 Conv_2_Relu_constraint_1[8,3]: - Conv_2_t[8,3] + Conv_2_Relu_ts[8,3]
   - Conv_2_Relu_ss[8,3] = 0
 Conv_2_Relu_constraint_1[8,4]: - Conv_2_t[8,4] + Conv_2_Relu_ts[8,4]
   - Conv_2_Relu_ss[8,4] = 0
 Conv_2_Relu_constraint_1[8,5]: - Conv_2_t[8,5] + Conv_2_Relu_ts[8,5]
   - Conv_2_Relu_ss[8,5] = 0
 Conv_2_Relu_constraint_1[8,6]: - Conv_2_t[8,6] + Conv_2_Relu_ts[8,6]
   - Conv_2_Relu_ss[8,6] = 0
 Conv_2_Relu_constraint_1[8,7]: - Conv_2_t[8,7] + Conv_2_Relu_ts[8,7]
   - Conv_2_Relu_ss[8,7] = 0
 Conv_2_Relu_constraint_1[9,0]: - Conv_2_t[9,0] + Conv_2_Relu_ts[9,0]
   - Conv_2_Relu_ss[9,0] = 0
 Conv_2_Relu_constraint_1[9,1]: - Conv_2_t[9,1] + Conv_2_Relu_ts[9,1]
   - Conv_2_Relu_ss[9,1] = 0
 Conv_2_Relu_constraint_1[9,2]: - Conv_2_t[9,2] + Conv_2_Relu_ts[9,2]
   - Conv_2_Relu_ss[9,2] = 0
 Conv_2_Relu_constraint_1[9,3]: - Conv_2_t[9,3] + Conv_2_Relu_ts[9,3]
   - Conv_2_Relu_ss[9,3] = 0
 Conv_2_Relu_constraint_1[9,4]: - Conv_2_t[9,4] + Conv_2_Relu_ts[9,4]
   - Conv_2_Relu_ss[9,4] = 0
 Conv_2_Relu_constraint_1[9,5]: - Conv_2_t[9,5] + Conv_2_Relu_ts[9,5]
   - Conv_2_Relu_ss[9,5] = 0
 Conv_2_Relu_constraint_1[9,6]: - Conv_2_t[9,6] + Conv_2_Relu_ts[9,6]
   - Conv_2_Relu_ss[9,6] = 0
 Conv_2_Relu_constraint_1[9,7]: - Conv_2_t[9,7] + Conv_2_Relu_ts[9,7]
   - Conv_2_Relu_ss[9,7] = 0
 Conv_2_Relu_constraint_2[0,0]: Conv_2_Relu_ts[0,0] - 256 C1960 <= 0
 Conv_2_Relu_constraint_2[0,1]: Conv_2_Relu_ts[0,1]
   - 126.8091092372764 C1961 <= 0
 Conv_2_Relu_constraint_2[0,2]: Conv_2_Relu_ts[0,2]
   - 252.2492312397784 C1962 <= 0
 Conv_2_Relu_constraint_2[0,3]: Conv_2_Relu_ts[0,3] - 256 C1963 <= 0
 Conv_2_Relu_constraint_2[0,4]: Conv_2_Relu_ts[0,4] - 256 C1964 <= 0
 Conv_2_Relu_constraint_2[0,5]: Conv_2_Relu_ts[0,5] - 256 C1965 <= 0
 Conv_2_Relu_constraint_2[0,6]: Conv_2_Relu_ts[0,6] - 256 C1966 <= 0
 Conv_2_Relu_constraint_2[0,7]: Conv_2_Relu_ts[0,7]
   - 11.47895694870232 C1967 <= 0
 Conv_2_Relu_constraint_2[1,0]: Conv_2_Relu_ts[1,0] - 256 C1968 <= 0
 Conv_2_Relu_constraint_2[1,1]: Conv_2_Relu_ts[1,1]
   - 126.8091092372764 C1969 <= 0
 Conv_2_Relu_constraint_2[1,2]: Conv_2_Relu_ts[1,2]
   - 252.2492312397784 C1970 <= 0
 Conv_2_Relu_constraint_2[1,3]: Conv_2_Relu_ts[1,3] - 256 C1971 <= 0
 Conv_2_Relu_constraint_2[1,4]: Conv_2_Relu_ts[1,4] - 256 C1972 <= 0
 Conv_2_Relu_constraint_2[1,5]: Conv_2_Relu_ts[1,5] - 256 C1973 <= 0
 Conv_2_Relu_constraint_2[1,6]: Conv_2_Relu_ts[1,6] - 256 C1974 <= 0
 Conv_2_Relu_constraint_2[1,7]: Conv_2_Relu_ts[1,7]
   - 11.47895694870232 C1975 <= 0
 Conv_2_Relu_constraint_2[2,0]: Conv_2_Relu_ts[2,0] - 256 C1976 <= 0
 Conv_2_Relu_constraint_2[2,1]: Conv_2_Relu_ts[2,1]
   - 126.8091092372764 C1977 <= 0
 Conv_2_Relu_constraint_2[2,2]: Conv_2_Relu_ts[2,2]
   - 252.2492312397784 C1978 <= 0
 Conv_2_Relu_constraint_2[2,3]: Conv_2_Relu_ts[2,3] - 256 C1979 <= 0
 Conv_2_Relu_constraint_2[2,4]: Conv_2_Relu_ts[2,4] - 256 C1980 <= 0
 Conv_2_Relu_constraint_2[2,5]: Conv_2_Relu_ts[2,5] - 256 C1981 <= 0
 Conv_2_Relu_constraint_2[2,6]: Conv_2_Relu_ts[2,6] - 256 C1982 <= 0
 Conv_2_Relu_constraint_2[2,7]: Conv_2_Relu_ts[2,7]
   - 11.47895694870232 C1983 <= 0
 Conv_2_Relu_constraint_2[3,0]: Conv_2_Relu_ts[3,0] - 256 C1984 <= 0
 Conv_2_Relu_constraint_2[3,1]: Conv_2_Relu_ts[3,1]
   - 126.8091092372764 C1985 <= 0
 Conv_2_Relu_constraint_2[3,2]: Conv_2_Relu_ts[3,2]
   - 252.2492312397784 C1986 <= 0
 Conv_2_Relu_constraint_2[3,3]: Conv_2_Relu_ts[3,3] - 256 C1987 <= 0
 Conv_2_Relu_constraint_2[3,4]: Conv_2_Relu_ts[3,4] - 256 C1988 <= 0
 Conv_2_Relu_constraint_2[3,5]: Conv_2_Relu_ts[3,5] - 256 C1989 <= 0
 Conv_2_Relu_constraint_2[3,6]: Conv_2_Relu_ts[3,6] - 256 C1990 <= 0
 Conv_2_Relu_constraint_2[3,7]: Conv_2_Relu_ts[3,7]
   - 11.47895694870232 C1991 <= 0
 Conv_2_Relu_constraint_2[4,0]: Conv_2_Relu_ts[4,0] - 256 C1992 <= 0
 Conv_2_Relu_constraint_2[4,1]: Conv_2_Relu_ts[4,1]
   - 126.8091092372764 C1993 <= 0
 Conv_2_Relu_constraint_2[4,2]: Conv_2_Relu_ts[4,2]
   - 252.2492312397784 C1994 <= 0
 Conv_2_Relu_constraint_2[4,3]: Conv_2_Relu_ts[4,3] - 256 C1995 <= 0
 Conv_2_Relu_constraint_2[4,4]: Conv_2_Relu_ts[4,4] - 256 C1996 <= 0
 Conv_2_Relu_constraint_2[4,5]: Conv_2_Relu_ts[4,5] - 256 C1997 <= 0
 Conv_2_Relu_constraint_2[4,6]: Conv_2_Relu_ts[4,6] - 256 C1998 <= 0
 Conv_2_Relu_constraint_2[4,7]: Conv_2_Relu_ts[4,7]
   - 11.47895694870232 C1999 <= 0
 Conv_2_Relu_constraint_2[5,0]: Conv_2_Relu_ts[5,0] - 256 C2000 <= 0
 Conv_2_Relu_constraint_2[5,1]: Conv_2_Relu_ts[5,1]
   - 126.8091092372764 C2001 <= 0
 Conv_2_Relu_constraint_2[5,2]: Conv_2_Relu_ts[5,2]
   - 252.2492312397784 C2002 <= 0
 Conv_2_Relu_constraint_2[5,3]: Conv_2_Relu_ts[5,3] - 256 C2003 <= 0
 Conv_2_Relu_constraint_2[5,4]: Conv_2_Relu_ts[5,4] - 256 C2004 <= 0
 Conv_2_Relu_constraint_2[5,5]: Conv_2_Relu_ts[5,5] - 256 C2005 <= 0
 Conv_2_Relu_constraint_2[5,6]: Conv_2_Relu_ts[5,6] - 256 C2006 <= 0
 Conv_2_Relu_constraint_2[5,7]: Conv_2_Relu_ts[5,7]
   - 11.47895694870232 C2007 <= 0
 Conv_2_Relu_constraint_2[6,0]: Conv_2_Relu_ts[6,0] - 256 C2008 <= 0
 Conv_2_Relu_constraint_2[6,1]: Conv_2_Relu_ts[6,1]
   - 126.8091092372764 C2009 <= 0
 Conv_2_Relu_constraint_2[6,2]: Conv_2_Relu_ts[6,2]
   - 252.2492312397784 C2010 <= 0
 Conv_2_Relu_constraint_2[6,3]: Conv_2_Relu_ts[6,3] - 256 C2011 <= 0
 Conv_2_Relu_constraint_2[6,4]: Conv_2_Relu_ts[6,4] - 256 C2012 <= 0
 Conv_2_Relu_constraint_2[6,5]: Conv_2_Relu_ts[6,5] - 256 C2013 <= 0
 Conv_2_Relu_constraint_2[6,6]: Conv_2_Relu_ts[6,6] - 256 C2014 <= 0
 Conv_2_Relu_constraint_2[6,7]: Conv_2_Relu_ts[6,7]
   - 11.47895694870232 C2015 <= 0
 Conv_2_Relu_constraint_2[7,0]: Conv_2_Relu_ts[7,0] - 256 C2016 <= 0
 Conv_2_Relu_constraint_2[7,1]: Conv_2_Relu_ts[7,1]
   - 126.8091092372764 C2017 <= 0
 Conv_2_Relu_constraint_2[7,2]: Conv_2_Relu_ts[7,2]
   - 252.2492312397784 C2018 <= 0
 Conv_2_Relu_constraint_2[7,3]: Conv_2_Relu_ts[7,3] - 256 C2019 <= 0
 Conv_2_Relu_constraint_2[7,4]: Conv_2_Relu_ts[7,4] - 256 C2020 <= 0
 Conv_2_Relu_constraint_2[7,5]: Conv_2_Relu_ts[7,5] - 256 C2021 <= 0
 Conv_2_Relu_constraint_2[7,6]: Conv_2_Relu_ts[7,6] - 256 C2022 <= 0
 Conv_2_Relu_constraint_2[7,7]: Conv_2_Relu_ts[7,7]
   - 11.47895694870232 C2023 <= 0
 Conv_2_Relu_constraint_2[8,0]: Conv_2_Relu_ts[8,0] - 256 C2024 <= 0
 Conv_2_Relu_constraint_2[8,1]: Conv_2_Relu_ts[8,1]
   - 126.8091092372764 C2025 <= 0
 Conv_2_Relu_constraint_2[8,2]: Conv_2_Relu_ts[8,2]
   - 252.2492312397784 C2026 <= 0
 Conv_2_Relu_constraint_2[8,3]: Conv_2_Relu_ts[8,3] - 256 C2027 <= 0
 Conv_2_Relu_constraint_2[8,4]: Conv_2_Relu_ts[8,4] - 256 C2028 <= 0
 Conv_2_Relu_constraint_2[8,5]: Conv_2_Relu_ts[8,5] - 256 C2029 <= 0
 Conv_2_Relu_constraint_2[8,6]: Conv_2_Relu_ts[8,6] - 256 C2030 <= 0
 Conv_2_Relu_constraint_2[8,7]: Conv_2_Relu_ts[8,7]
   - 11.47895694870232 C2031 <= 0
 Conv_2_Relu_constraint_2[9,0]: Conv_2_Relu_ts[9,0] - 256 C2032 <= 0
 Conv_2_Relu_constraint_2[9,1]: Conv_2_Relu_ts[9,1]
   - 126.8091092372764 C2033 <= 0
 Conv_2_Relu_constraint_2[9,2]: Conv_2_Relu_ts[9,2]
   - 252.2492312397784 C2034 <= 0
 Conv_2_Relu_constraint_2[9,3]: Conv_2_Relu_ts[9,3] - 256 C2035 <= 0
 Conv_2_Relu_constraint_2[9,4]: Conv_2_Relu_ts[9,4] - 256 C2036 <= 0
 Conv_2_Relu_constraint_2[9,5]: Conv_2_Relu_ts[9,5] - 256 C2037 <= 0
 Conv_2_Relu_constraint_2[9,6]: Conv_2_Relu_ts[9,6] - 256 C2038 <= 0
 Conv_2_Relu_constraint_2[9,7]: Conv_2_Relu_ts[9,7]
   - 11.47895694870232 C2039 <= 0
 Conv_2_Relu_constraint_3[0,0]: Conv_2_Relu_ss[0,0] + 256 C1960 <= 256
 Conv_2_Relu_constraint_3[0,1]: Conv_2_Relu_ss[0,1] + 256 C1961 <= 256
 Conv_2_Relu_constraint_3[0,2]: Conv_2_Relu_ss[0,2] + 256 C1962 <= 256
 Conv_2_Relu_constraint_3[0,3]: Conv_2_Relu_ss[0,3] + 256 C1963 <= 256
 Conv_2_Relu_constraint_3[0,4]: Conv_2_Relu_ss[0,4]
   + 242.8364949340469 C1964 <= 242.8364949340469
 Conv_2_Relu_constraint_3[0,5]: Conv_2_Relu_ss[0,5] + 256 C1965 <= 256
 Conv_2_Relu_constraint_3[0,6]: Conv_2_Relu_ss[0,6]
   + 96.48911034049161 C1966 <= 96.48911034049161
 Conv_2_Relu_constraint_3[0,7]: Conv_2_Relu_ss[0,7]
   + 101.2542460305265 C1967 <= 101.2542460305265
 Conv_2_Relu_constraint_3[1,0]: Conv_2_Relu_ss[1,0] + 256 C1968 <= 256
 Conv_2_Relu_constraint_3[1,1]: Conv_2_Relu_ss[1,1] + 256 C1969 <= 256
 Conv_2_Relu_constraint_3[1,2]: Conv_2_Relu_ss[1,2] + 256 C1970 <= 256
 Conv_2_Relu_constraint_3[1,3]: Conv_2_Relu_ss[1,3] + 256 C1971 <= 256
 Conv_2_Relu_constraint_3[1,4]: Conv_2_Relu_ss[1,4]
   + 242.8364949340469 C1972 <= 242.8364949340469
 Conv_2_Relu_constraint_3[1,5]: Conv_2_Relu_ss[1,5] + 256 C1973 <= 256
 Conv_2_Relu_constraint_3[1,6]: Conv_2_Relu_ss[1,6]
   + 96.48911034049161 C1974 <= 96.48911034049161
 Conv_2_Relu_constraint_3[1,7]: Conv_2_Relu_ss[1,7]
   + 101.2542460305265 C1975 <= 101.2542460305265
 Conv_2_Relu_constraint_3[2,0]: Conv_2_Relu_ss[2,0] + 256 C1976 <= 256
 Conv_2_Relu_constraint_3[2,1]: Conv_2_Relu_ss[2,1] + 256 C1977 <= 256
 Conv_2_Relu_constraint_3[2,2]: Conv_2_Relu_ss[2,2] + 256 C1978 <= 256
 Conv_2_Relu_constraint_3[2,3]: Conv_2_Relu_ss[2,3] + 256 C1979 <= 256
 Conv_2_Relu_constraint_3[2,4]: Conv_2_Relu_ss[2,4]
   + 242.8364949340469 C1980 <= 242.8364949340469
 Conv_2_Relu_constraint_3[2,5]: Conv_2_Relu_ss[2,5] + 256 C1981 <= 256
 Conv_2_Relu_constraint_3[2,6]: Conv_2_Relu_ss[2,6]
   + 96.48911034049161 C1982 <= 96.48911034049161
 Conv_2_Relu_constraint_3[2,7]: Conv_2_Relu_ss[2,7]
   + 101.2542460305265 C1983 <= 101.2542460305265
 Conv_2_Relu_constraint_3[3,0]: Conv_2_Relu_ss[3,0] + 256 C1984 <= 256
 Conv_2_Relu_constraint_3[3,1]: Conv_2_Relu_ss[3,1] + 256 C1985 <= 256
 Conv_2_Relu_constraint_3[3,2]: Conv_2_Relu_ss[3,2] + 256 C1986 <= 256
 Conv_2_Relu_constraint_3[3,3]: Conv_2_Relu_ss[3,3] + 256 C1987 <= 256
 Conv_2_Relu_constraint_3[3,4]: Conv_2_Relu_ss[3,4]
   + 242.8364949340469 C1988 <= 242.8364949340469
 Conv_2_Relu_constraint_3[3,5]: Conv_2_Relu_ss[3,5] + 256 C1989 <= 256
 Conv_2_Relu_constraint_3[3,6]: Conv_2_Relu_ss[3,6]
   + 96.48911034049161 C1990 <= 96.48911034049161
 Conv_2_Relu_constraint_3[3,7]: Conv_2_Relu_ss[3,7]
   + 101.2542460305265 C1991 <= 101.2542460305265
 Conv_2_Relu_constraint_3[4,0]: Conv_2_Relu_ss[4,0] + 256 C1992 <= 256
 Conv_2_Relu_constraint_3[4,1]: Conv_2_Relu_ss[4,1] + 256 C1993 <= 256
 Conv_2_Relu_constraint_3[4,2]: Conv_2_Relu_ss[4,2] + 256 C1994 <= 256
 Conv_2_Relu_constraint_3[4,3]: Conv_2_Relu_ss[4,3] + 256 C1995 <= 256
 Conv_2_Relu_constraint_3[4,4]: Conv_2_Relu_ss[4,4]
   + 242.8364949340469 C1996 <= 242.8364949340469
 Conv_2_Relu_constraint_3[4,5]: Conv_2_Relu_ss[4,5] + 256 C1997 <= 256
 Conv_2_Relu_constraint_3[4,6]: Conv_2_Relu_ss[4,6]
   + 96.48911034049161 C1998 <= 96.48911034049161
 Conv_2_Relu_constraint_3[4,7]: Conv_2_Relu_ss[4,7]
   + 101.2542460305265 C1999 <= 101.2542460305265
 Conv_2_Relu_constraint_3[5,0]: Conv_2_Relu_ss[5,0] + 256 C2000 <= 256
 Conv_2_Relu_constraint_3[5,1]: Conv_2_Relu_ss[5,1] + 256 C2001 <= 256
 Conv_2_Relu_constraint_3[5,2]: Conv_2_Relu_ss[5,2] + 256 C2002 <= 256
 Conv_2_Relu_constraint_3[5,3]: Conv_2_Relu_ss[5,3] + 256 C2003 <= 256
 Conv_2_Relu_constraint_3[5,4]: Conv_2_Relu_ss[5,4]
   + 242.8364949340469 C2004 <= 242.8364949340469
 Conv_2_Relu_constraint_3[5,5]: Conv_2_Relu_ss[5,5] + 256 C2005 <= 256
 Conv_2_Relu_constraint_3[5,6]: Conv_2_Relu_ss[5,6]
   + 96.48911034049161 C2006 <= 96.48911034049161
 Conv_2_Relu_constraint_3[5,7]: Conv_2_Relu_ss[5,7]
   + 101.2542460305265 C2007 <= 101.2542460305265
 Conv_2_Relu_constraint_3[6,0]: Conv_2_Relu_ss[6,0] + 256 C2008 <= 256
 Conv_2_Relu_constraint_3[6,1]: Conv_2_Relu_ss[6,1] + 256 C2009 <= 256
 Conv_2_Relu_constraint_3[6,2]: Conv_2_Relu_ss[6,2] + 256 C2010 <= 256
 Conv_2_Relu_constraint_3[6,3]: Conv_2_Relu_ss[6,3] + 256 C2011 <= 256
 Conv_2_Relu_constraint_3[6,4]: Conv_2_Relu_ss[6,4]
   + 242.8364949340469 C2012 <= 242.8364949340469
 Conv_2_Relu_constraint_3[6,5]: Conv_2_Relu_ss[6,5] + 256 C2013 <= 256
 Conv_2_Relu_constraint_3[6,6]: Conv_2_Relu_ss[6,6]
   + 96.48911034049161 C2014 <= 96.48911034049161
 Conv_2_Relu_constraint_3[6,7]: Conv_2_Relu_ss[6,7]
   + 101.2542460305265 C2015 <= 101.2542460305265
 Conv_2_Relu_constraint_3[7,0]: Conv_2_Relu_ss[7,0] + 256 C2016 <= 256
 Conv_2_Relu_constraint_3[7,1]: Conv_2_Relu_ss[7,1] + 256 C2017 <= 256
 Conv_2_Relu_constraint_3[7,2]: Conv_2_Relu_ss[7,2] + 256 C2018 <= 256
 Conv_2_Relu_constraint_3[7,3]: Conv_2_Relu_ss[7,3] + 256 C2019 <= 256
 Conv_2_Relu_constraint_3[7,4]: Conv_2_Relu_ss[7,4]
   + 242.8364949340469 C2020 <= 242.8364949340469
 Conv_2_Relu_constraint_3[7,5]: Conv_2_Relu_ss[7,5] + 256 C2021 <= 256
 Conv_2_Relu_constraint_3[7,6]: Conv_2_Relu_ss[7,6]
   + 96.48911034049161 C2022 <= 96.48911034049161
 Conv_2_Relu_constraint_3[7,7]: Conv_2_Relu_ss[7,7]
   + 101.2542460305265 C2023 <= 101.2542460305265
 Conv_2_Relu_constraint_3[8,0]: Conv_2_Relu_ss[8,0] + 256 C2024 <= 256
 Conv_2_Relu_constraint_3[8,1]: Conv_2_Relu_ss[8,1] + 256 C2025 <= 256
 Conv_2_Relu_constraint_3[8,2]: Conv_2_Relu_ss[8,2] + 256 C2026 <= 256
 Conv_2_Relu_constraint_3[8,3]: Conv_2_Relu_ss[8,3] + 256 C2027 <= 256
 Conv_2_Relu_constraint_3[8,4]: Conv_2_Relu_ss[8,4]
   + 242.8364949340469 C2028 <= 242.8364949340469
 Conv_2_Relu_constraint_3[8,5]: Conv_2_Relu_ss[8,5] + 256 C2029 <= 256
 Conv_2_Relu_constraint_3[8,6]: Conv_2_Relu_ss[8,6]
   + 96.48911034049161 C2030 <= 96.48911034049161
 Conv_2_Relu_constraint_3[8,7]: Conv_2_Relu_ss[8,7]
   + 101.2542460305265 C2031 <= 101.2542460305265
 Conv_2_Relu_constraint_3[9,0]: Conv_2_Relu_ss[9,0] + 256 C2032 <= 256
 Conv_2_Relu_constraint_3[9,1]: Conv_2_Relu_ss[9,1] + 256 C2033 <= 256
 Conv_2_Relu_constraint_3[9,2]: Conv_2_Relu_ss[9,2] + 256 C2034 <= 256
 Conv_2_Relu_constraint_3[9,3]: Conv_2_Relu_ss[9,3] + 256 C2035 <= 256
 Conv_2_Relu_constraint_3[9,4]: Conv_2_Relu_ss[9,4]
   + 242.8364949340469 C2036 <= 242.8364949340469
 Conv_2_Relu_constraint_3[9,5]: Conv_2_Relu_ss[9,5] + 256 C2037 <= 256
 Conv_2_Relu_constraint_3[9,6]: Conv_2_Relu_ss[9,6]
   + 96.48911034049161 C2038 <= 96.48911034049161
 Conv_2_Relu_constraint_3[9,7]: Conv_2_Relu_ss[9,7]
   + 101.2542460305265 C2039 <= 101.2542460305265
 Aggregation_constraint[0]: - 0.1 Conv_2_Relu_ts[0,0]
   - 0.1 Conv_2_Relu_ts[1,0] - 0.1 Conv_2_Relu_ts[2,0]
   - 0.1 Conv_2_Relu_ts[3,0] - 0.1 Conv_2_Relu_ts[4,0]
   - 0.1 Conv_2_Relu_ts[5,0] - 0.1 Conv_2_Relu_ts[6,0]
   - 0.1 Conv_2_Relu_ts[7,0] - 0.1 Conv_2_Relu_ts[8,0]
   - 0.1 Conv_2_Relu_ts[9,0] + Aggregation[0] = 0
 Aggregation_constraint[1]: - 0.1 Conv_2_Relu_ts[0,1]
   - 0.1 Conv_2_Relu_ts[1,1] - 0.1 Conv_2_Relu_ts[2,1]
   - 0.1 Conv_2_Relu_ts[3,1] - 0.1 Conv_2_Relu_ts[4,1]
   - 0.1 Conv_2_Relu_ts[5,1] - 0.1 Conv_2_Relu_ts[6,1]
   - 0.1 Conv_2_Relu_ts[7,1] - 0.1 Conv_2_Relu_ts[8,1]
   - 0.1 Conv_2_Relu_ts[9,1] + Aggregation[1] = 0
 Aggregation_constraint[2]: - 0.1 Conv_2_Relu_ts[0,2]
   - 0.1 Conv_2_Relu_ts[1,2] - 0.1 Conv_2_Relu_ts[2,2]
   - 0.1 Conv_2_Relu_ts[3,2] - 0.1 Conv_2_Relu_ts[4,2]
   - 0.1 Conv_2_Relu_ts[5,2] - 0.1 Conv_2_Relu_ts[6,2]
   - 0.1 Conv_2_Relu_ts[7,2] - 0.1 Conv_2_Relu_ts[8,2]
   - 0.1 Conv_2_Relu_ts[9,2] + Aggregation[2] = 0
 Aggregation_constraint[3]: - 0.1 Conv_2_Relu_ts[0,3]
   - 0.1 Conv_2_Relu_ts[1,3] - 0.1 Conv_2_Relu_ts[2,3]
   - 0.1 Conv_2_Relu_ts[3,3] - 0.1 Conv_2_Relu_ts[4,3]
   - 0.1 Conv_2_Relu_ts[5,3] - 0.1 Conv_2_Relu_ts[6,3]
   - 0.1 Conv_2_Relu_ts[7,3] - 0.1 Conv_2_Relu_ts[8,3]
   - 0.1 Conv_2_Relu_ts[9,3] + Aggregation[3] = 0
 Aggregation_constraint[4]: - 0.1 Conv_2_Relu_ts[0,4]
   - 0.1 Conv_2_Relu_ts[1,4] - 0.1 Conv_2_Relu_ts[2,4]
   - 0.1 Conv_2_Relu_ts[3,4] - 0.1 Conv_2_Relu_ts[4,4]
   - 0.1 Conv_2_Relu_ts[5,4] - 0.1 Conv_2_Relu_ts[6,4]
   - 0.1 Conv_2_Relu_ts[7,4] - 0.1 Conv_2_Relu_ts[8,4]
   - 0.1 Conv_2_Relu_ts[9,4] + Aggregation[4] = 0
 Aggregation_constraint[5]: - 0.1 Conv_2_Relu_ts[0,5]
   - 0.1 Conv_2_Relu_ts[1,5] - 0.1 Conv_2_Relu_ts[2,5]
   - 0.1 Conv_2_Relu_ts[3,5] - 0.1 Conv_2_Relu_ts[4,5]
   - 0.1 Conv_2_Relu_ts[5,5] - 0.1 Conv_2_Relu_ts[6,5]
   - 0.1 Conv_2_Relu_ts[7,5] - 0.1 Conv_2_Relu_ts[8,5]
   - 0.1 Conv_2_Relu_ts[9,5] + Aggregation[5] = 0
 Aggregation_constraint[6]: - 0.1 Conv_2_Relu_ts[0,6]
   - 0.1 Conv_2_Relu_ts[1,6] - 0.1 Conv_2_Relu_ts[2,6]
   - 0.1 Conv_2_Relu_ts[3,6] - 0.1 Conv_2_Relu_ts[4,6]
   - 0.1 Conv_2_Relu_ts[5,6] - 0.1 Conv_2_Relu_ts[6,6]
   - 0.1 Conv_2_Relu_ts[7,6] - 0.1 Conv_2_Relu_ts[8,6]
   - 0.1 Conv_2_Relu_ts[9,6] + Aggregation[6] = 0
 Aggregation_constraint[7]: - 0.1 Conv_2_Relu_ts[0,7]
   - 0.1 Conv_2_Relu_ts[1,7] - 0.1 Conv_2_Relu_ts[2,7]
   - 0.1 Conv_2_Relu_ts[3,7] - 0.1 Conv_2_Relu_ts[4,7]
   - 0.1 Conv_2_Relu_ts[5,7] - 0.1 Conv_2_Relu_ts[6,7]
   - 0.1 Conv_2_Relu_ts[7,7] - 0.1 Conv_2_Relu_ts[8,7]
   - 0.1 Conv_2_Relu_ts[9,7] + Aggregation[7] = 0
 Lin_0_output_constraint[0,0]: 0.6212125420570374 Aggregation[0]
   - 0.5678613781929016 Aggregation[1] + 5.249436855316162 Aggregation[2]
   - 0.734696090221405 Aggregation[3] + 0.9422561526298523 Aggregation[4]
   + 0.6638888120651245 Aggregation[5] + 1.270898699760437 Aggregation[6]
   - 0.1612947434186935 Aggregation[7] + Lin_0_t[0] = -0.1502677947282791
 Lin_0_output_constraint[0,1]: 0.5056337118148804 Aggregation[0]
   + 0.3998594582080841 Aggregation[1] - 1.443193912506104 Aggregation[2]
   + 1.501495718955994 Aggregation[3] + 0.1535968035459518 Aggregation[4]
   + 0.3860666751861572 Aggregation[5] + 1.570986747741699 Aggregation[6]
   - 0.1883221119642258 Aggregation[7] + Lin_0_t[1] = -0.1219431757926941
 Lin_0_output_constraint[0,2]: - 2.031019926071167 Aggregation[0]
   - 1.240094184875488 Aggregation[1] + 1.392327666282654 Aggregation[2]
   + 0.1871164441108704 Aggregation[3] + 0.9084514379501343 Aggregation[4]
   + 0.1870955526828766 Aggregation[5] + 1.100360870361328 Aggregation[6]
   + 0.1981218159198761 Aggregation[7] + Lin_0_t[2] = -1.560397744178772
 Lin_0_output_constraint[0,3]: 0.0826235264539719 Aggregation[0]
   - 0.1184611096978188 Aggregation[1] - 0.00286348070949316 Aggregation[2]
   + 0.4431280493736267 Aggregation[3] + 0.3206478953361511 Aggregation[4]
   + 0.136445090174675 Aggregation[5] + 0.2083217054605484 Aggregation[6]
   + 0.3595787286758423 Aggregation[7] + Lin_0_t[3] = -0.2082993686199188
 Lin_0_output_constraint[0,4]: 0.3427026271820068 Aggregation[0]
   - 0.7109329700469971 Aggregation[1] - 0.793476939201355 Aggregation[2]
   + 1.773179173469543 Aggregation[3] + 2.093883991241455 Aggregation[4]
   - 1.053140640258789 Aggregation[5] - 0.1948821991682053 Aggregation[6]
   + 0.2688142061233521 Aggregation[7] + Lin_0_t[4] = 0.1053846329450607
 Lin_0_output_constraint[0,5]: - 0.895245373249054 Aggregation[0]
   + 0.3303574025630951 Aggregation[1] - 0.0889761596918106 Aggregation[2]
   - 0.8579750657081604 Aggregation[3] - 2.582921743392944 Aggregation[4]
   + 0.2820542752742767 Aggregation[5] - 0.6246813535690308 Aggregation[6]
   - 0.1395417451858521 Aggregation[7] + Lin_0_t[5] = -1.053776383399963
 Lin_0_output_constraint[0,6]: - 2.07094144821167 Aggregation[0]
   - 0.2997556328773499 Aggregation[1] - 0.1103238686919212 Aggregation[2]
   + 2.739991426467896 Aggregation[3] + 0.2204796075820923 Aggregation[4]
   + 0.2711255550384521 Aggregation[5] - 0.0743993371725082 Aggregation[6]
   + 0.2743711769580841 Aggregation[7] + Lin_0_t[6] = 2.432949781417847
 Lin_0_output_constraint[0,7]: 3.304619312286377 Aggregation[0]
   + 3.144479751586914 Aggregation[1] + 2.009815692901611 Aggregation[2]
   - 0.1087105572223663 Aggregation[3] - 0.7821275591850281 Aggregation[4]
   + 2.172999143600464 Aggregation[5] + 5.288445472717285 Aggregation[6]
   - 0.3791896402835846 Aggregation[7] + Lin_0_t[7] = 1.161705017089844
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
 Lin_0_Relu_constraint_2[0,0]: Lin_0_Relu_ts[0,0] - 256 C2072 <= 0
 Lin_0_Relu_constraint_2[0,1]: Lin_0_Relu_ts[0,1] - 256 C2073 <= 0
 Lin_0_Relu_constraint_2[0,2]: Lin_0_Relu_ts[0,2] - 256 C2074 <= 0
 Lin_0_Relu_constraint_2[0,3]: Lin_0_Relu_ts[0,3] - 15.74425860767926 C2075
   <= 0
 Lin_0_Relu_constraint_2[0,4]: Lin_0_Relu_ts[0,4] - 256 C2076 <= 0
 Lin_0_Relu_constraint_2[0,5]: Lin_0_Relu_ts[0,5] - 256 C2077 <= 0
 Lin_0_Relu_constraint_2[0,6]: Lin_0_Relu_ts[0,6] - 256 C2078 <= 0
 Lin_0_Relu_constraint_2[0,7]: Lin_0_Relu_ts[0,7] - 232.4072593565022 C2079
   <= 0
 Lin_0_Relu_constraint_3[0,0]: Lin_0_Relu_ss[0,0] + 256 C2072 <= 256
 Lin_0_Relu_constraint_3[0,1]: Lin_0_Relu_ss[0,1] + 256 C2073 <= 256
 Lin_0_Relu_constraint_3[0,2]: Lin_0_Relu_ss[0,2] + 256 C2074 <= 256
 Lin_0_Relu_constraint_3[0,3]: Lin_0_Relu_ss[0,3] + 256 C2075 <= 256
 Lin_0_Relu_constraint_3[0,4]: Lin_0_Relu_ss[0,4] + 256 C2076 <= 256
 Lin_0_Relu_constraint_3[0,5]: Lin_0_Relu_ss[0,5] + 114.0982224191812 C2077
   <= 114.0982224191812
 Lin_0_Relu_constraint_3[0,6]: Lin_0_Relu_ss[0,6] + 256 C2078 <= 256
 Lin_0_Relu_constraint_3[0,7]: Lin_0_Relu_ss[0,7] + 256 C2079 <= 256
 Lin_1_output_constraint[0,0]: - 0.5132390856742859 Lin_0_Relu_ts[0,0]
   - 0.2247032225131989 Lin_0_Relu_ts[0,1]
   + 0.7689714431762695 Lin_0_Relu_ts[0,2]
   + 0.2430446743965149 Lin_0_Relu_ts[0,3]
   - 1.275024056434631 Lin_0_Relu_ts[0,4]
   + 1.04903519153595 Lin_0_Relu_ts[0,5]
   - 0.4497078955173492 Lin_0_Relu_ts[0,6]
   - 0.6977813839912415 Lin_0_Relu_ts[0,7] + Lin_1_t[0] = 3.26350474357605
 Lin_1_output_constraint[0,1]: - 0.1767696142196655 Lin_0_Relu_ts[0,0]
   - 0.2787802815437317 Lin_0_Relu_ts[0,1]
   + 0.1111363023519516 Lin_0_Relu_ts[0,2]
   + 0.0319719240069389 Lin_0_Relu_ts[0,3]
   + 0.3975845873355865 Lin_0_Relu_ts[0,4]
   + 0.1960760504007339 Lin_0_Relu_ts[0,5]
   + 0.2229146957397461 Lin_0_Relu_ts[0,6]
   + 0.3091910779476166 Lin_0_Relu_ts[0,7] + Lin_1_t[1]
   = -0.2507714033126831
 Lin_1_output_constraint[0,2]: 0.484831690788269 Lin_0_Relu_ts[0,0]
   - 1.393808960914612 Lin_0_Relu_ts[0,1]
   - 0.2643370628356934 Lin_0_Relu_ts[0,2]
   - 0.0584821254014969 Lin_0_Relu_ts[0,3]
   + 0.6038019061088562 Lin_0_Relu_ts[0,4]
   + 1.646676540374756 Lin_0_Relu_ts[0,5]
   - 0.8018991947174072 Lin_0_Relu_ts[0,6]
   - 1.597251057624817 Lin_0_Relu_ts[0,7] + Lin_1_t[2] = 1.006668329238892
 Lin_1_output_constraint[0,3]: 0.8437172174453735 Lin_0_Relu_ts[0,0]
   - 0.0745344087481499 Lin_0_Relu_ts[0,1]
   + 1.785393714904785 Lin_0_Relu_ts[0,2]
   - 0.2477732002735138 Lin_0_Relu_ts[0,3]
   - 1.153511762619019 Lin_0_Relu_ts[0,4]
   - 0.3594806492328644 Lin_0_Relu_ts[0,5]
   + 1.882448196411133 Lin_0_Relu_ts[0,6]
   + 1.670949816703796 Lin_0_Relu_ts[0,7] + Lin_1_t[3]
   = -0.5804635286331177
 Lin_1_output_constraint[0,4]: 0.14272241294384 Lin_0_Relu_ts[0,0]
   - 0.00117842585314065 Lin_0_Relu_ts[0,1]
   + 0.3075129687786102 Lin_0_Relu_ts[0,2]
   - 0.0786379277706146 Lin_0_Relu_ts[0,3]
   + 0.3147511780261993 Lin_0_Relu_ts[0,4]
   + 0.3199646770954132 Lin_0_Relu_ts[0,5]
   + 0.2560481131076813 Lin_0_Relu_ts[0,6]
   + 0.0102797849103808 Lin_0_Relu_ts[0,7] + Lin_1_t[4]
   = -0.0813731327652931
 Lin_1_output_constraint[0,5]: 0.5963951945304871 Lin_0_Relu_ts[0,0]
   - 1.204570412635803 Lin_0_Relu_ts[0,1]
   - 1.873059868812561 Lin_0_Relu_ts[0,2]
   + 0.1828738749027252 Lin_0_Relu_ts[0,3]
   - 0.6438740491867065 Lin_0_Relu_ts[0,4]
   - 0.4487688839435577 Lin_0_Relu_ts[0,5]
   - 1.789048790931702 Lin_0_Relu_ts[0,6]
   + 1.058366179466248 Lin_0_Relu_ts[0,7] + Lin_1_t[5] = -3.06627345085144
 Lin_1_output_constraint[0,6]: - 0.1540552526712418 Lin_0_Relu_ts[0,0]
   - 0.2235594838857651 Lin_0_Relu_ts[0,1]
   + 0.2757455706596375 Lin_0_Relu_ts[0,2]
   - 0.1021358892321587 Lin_0_Relu_ts[0,3]
   + 0.3437762856483459 Lin_0_Relu_ts[0,4]
   + 0.2414004802703857 Lin_0_Relu_ts[0,5]
   + 0.2170199900865555 Lin_0_Relu_ts[0,6]
   + 0.1233204081654549 Lin_0_Relu_ts[0,7] + Lin_1_t[6]
   = -0.2105136513710022
 Lin_1_output_constraint[0,7]: 0.8264607191085815 Lin_0_Relu_ts[0,0]
   - 0.8349005579948425 Lin_0_Relu_ts[0,1]
   + 0.5134390592575073 Lin_0_Relu_ts[0,2]
   + 0.1891922056674957 Lin_0_Relu_ts[0,3]
   - 1.007278800010681 Lin_0_Relu_ts[0,4]
   - 0.6751555800437927 Lin_0_Relu_ts[0,5]
   - 0.0391219891607761 Lin_0_Relu_ts[0,6]
   + 2.510300159454346 Lin_0_Relu_ts[0,7] + Lin_1_t[7] = 4.353874206542969
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
 Lin1Relu_constraint_2[0,0]: Lin1Relu_ts[0,0] - 256 C2104 <= 0
 Lin1Relu_constraint_2[0,1]: Lin1Relu_ts[0,1] - 116.6207733154297 C2105
   <= 0
 Lin1Relu_constraint_2[0,2]: Lin1Relu_ts[0,2] - 256 C2106 <= 0
 Lin1Relu_constraint_2[0,3]: Lin1Relu_ts[0,3] - 256 C2107 <= 0
 Lin1Relu_constraint_2[0,4]: Lin1Relu_ts[0,4] - 1.539772889596566 C2108
   <= 0
 Lin1Relu_constraint_2[0,5]: Lin1Relu_ts[0,5] - 256 C2109 <= 0
 Lin1Relu_constraint_2[0,6]: Lin1Relu_ts[0,6] - 98.27742641179015 C2110
   <= 0
 Lin1Relu_constraint_2[0,7]: Lin1Relu_ts[0,7] - 256 C2111 <= 0
 Lin1Relu_constraint_3[0,0]: Lin1Relu_ss[0,0] + 256 C2104 <= 256
 Lin1Relu_constraint_3[0,1]: Lin1Relu_ss[0,1] + 256 C2105 <= 256
 Lin1Relu_constraint_3[0,2]: Lin1Relu_ss[0,2] + 256 C2106 <= 256
 Lin1Relu_constraint_3[0,3]: Lin1Relu_ss[0,3] + 256 C2107 <= 256
 Lin1Relu_constraint_3[0,4]: Lin1Relu_ss[0,4] + 256 C2108 <= 256
 Lin1Relu_constraint_3[0,5]: Lin1Relu_ss[0,5] + 256 C2109 <= 256
 Lin1Relu_constraint_3[0,6]: Lin1Relu_ss[0,6] + 256 C2110 <= 256
 Lin1Relu_constraint_3[0,7]: Lin1Relu_ss[0,7] + 256 C2111 <= 256
 Output_output_constraint[0,0]: 1.994431853294373 Lin1Relu_ts[0,0]
   + 0.0364780053496361 Lin1Relu_ts[0,1]
   + 1.158305168151855 Lin1Relu_ts[0,2]
   - 0.5776956081390381 Lin1Relu_ts[0,3]
   - 0.094692587852478 Lin1Relu_ts[0,4]
   + 1.863028049468994 Lin1Relu_ts[0,5] + 0.22263103723526 Lin1Relu_ts[0,6]
   - 0.7304947376251221 Lin1Relu_ts[0,7] + Output_t[0]
   = 0.6862915754318237
 Output_output_constraint[0,1]: - 0.9064277410507202 Lin1Relu_ts[0,0]
   + 0.2144530415534973 Lin1Relu_ts[0,1]
   - 1.065591216087341 Lin1Relu_ts[0,2]
   - 0.3800874948501587 Lin1Relu_ts[0,3]
   - 0.2792306542396545 Lin1Relu_ts[0,4]
   + 0.4052551984786987 Lin1Relu_ts[0,5]
   + 0.3277643024921417 Lin1Relu_ts[0,6]
   + 0.6476028561592102 Lin1Relu_ts[0,7] + Output_t[1]
   = -0.3277720212936401
 Output_output_constraint[0,2]: - 1.414348959922791 Lin1Relu_ts[0,0]
   - 0.1862725168466568 Lin1Relu_ts[0,1]
   + 1.257346749305725 Lin1Relu_ts[0,2]
   + 2.848361015319824 Lin1Relu_ts[0,3]
   + 0.0303341820836067 Lin1Relu_ts[0,4]
   + 3.36863899230957 Lin1Relu_ts[0,5]
   - 0.2299896478652954 Lin1Relu_ts[0,6]
   + 0.040352575480938 Lin1Relu_ts[0,7] + Output_t[2] = 3.606989860534668
 Output_output_constraint[0,3]: - 0.2113847136497498 Lin1Relu_ts[0,0]
   + 0.2599415183067322 Lin1Relu_ts[0,1]
   + 1.92596423625946 Lin1Relu_ts[0,2] + 1.227152943611145 Lin1Relu_ts[0,3]
   + 0.2704299092292786 Lin1Relu_ts[0,4]
   - 1.330726146697998 Lin1Relu_ts[0,5]
   + 0.1160476133227348 Lin1Relu_ts[0,6]
   + 2.498226642608643 Lin1Relu_ts[0,7] + Output_t[3] = -6.130110740661621
 Output_output_constraint[0,4]: 0.5598644018173218 Lin1Relu_ts[0,0]
   + 0.1113243922591209 Lin1Relu_ts[0,1]
   - 0.3955929279327393 Lin1Relu_ts[0,2]
   + 0.2518324255943298 Lin1Relu_ts[0,3]
   + 0.237128958106041 Lin1Relu_ts[0,4]
   + 0.4088925123214722 Lin1Relu_ts[0,5]
   - 0.0996533408761024 Lin1Relu_ts[0,6]
   - 0.4139194488525391 Lin1Relu_ts[0,7] + Output_t[4]
   = -1.437041401863098
 Conv_0_averages_constraint[0,0]: [ - A[0,0] * X[0,0]
   + A[0,0] * Conv_0_aggregated_features[0,0] - A[0,1] * X[1,0]
   - A[0,2] * X[2,0] - A[0,3] * X[3,0] - A[0,4] * X[4,0] - A[0,5] * X[5,0]
   - A[0,6] * X[6,0] - A[0,7] * X[7,0] - A[0,8] * X[8,0] - A[0,9] * X[9,0]
   + A[1,0] * Conv_0_aggregated_features[0,0]
   + A[2,0] * Conv_0_aggregated_features[0,0]
   + A[3,0] * Conv_0_aggregated_features[0,0]
   + A[4,0] * Conv_0_aggregated_features[0,0]
   + A[5,0] * Conv_0_aggregated_features[0,0]
   + A[6,0] * Conv_0_aggregated_features[0,0]
   + A[7,0] * Conv_0_aggregated_features[0,0]
   + A[8,0] * Conv_0_aggregated_features[0,0]
   + A[9,0] * Conv_0_aggregated_features[0,0] ] = 0
 Conv_0_averages_constraint[1,0]: [
   A[0,1] * Conv_0_aggregated_features[1,0] - A[1,0] * X[0,0]
   - A[1,1] * X[1,0] + A[1,1] * Conv_0_aggregated_features[1,0]
   - A[1,2] * X[2,0] - A[1,3] * X[3,0] - A[1,4] * X[4,0] - A[1,5] * X[5,0]
   - A[1,6] * X[6,0] - A[1,7] * X[7,0] - A[1,8] * X[8,0] - A[1,9] * X[9,0]
   + A[2,1] * Conv_0_aggregated_features[1,0]
   + A[3,1] * Conv_0_aggregated_features[1,0]
   + A[4,1] * Conv_0_aggregated_features[1,0]
   + A[5,1] * Conv_0_aggregated_features[1,0]
   + A[6,1] * Conv_0_aggregated_features[1,0]
   + A[7,1] * Conv_0_aggregated_features[1,0]
   + A[8,1] * Conv_0_aggregated_features[1,0]
   + A[9,1] * Conv_0_aggregated_features[1,0] ] = 0
 Conv_0_averages_constraint[2,0]: [
   A[0,2] * Conv_0_aggregated_features[2,0]
   + A[1,2] * Conv_0_aggregated_features[2,0] - A[2,0] * X[0,0]
   - A[2,1] * X[1,0] - A[2,2] * X[2,0]
   + A[2,2] * Conv_0_aggregated_features[2,0] - A[2,3] * X[3,0]
   - A[2,4] * X[4,0] - A[2,5] * X[5,0] - A[2,6] * X[6,0] - A[2,7] * X[7,0]
   - A[2,8] * X[8,0] - A[2,9] * X[9,0]
   + A[3,2] * Conv_0_aggregated_features[2,0]
   + A[4,2] * Conv_0_aggregated_features[2,0]
   + A[5,2] * Conv_0_aggregated_features[2,0]
   + A[6,2] * Conv_0_aggregated_features[2,0]
   + A[7,2] * Conv_0_aggregated_features[2,0]
   + A[8,2] * Conv_0_aggregated_features[2,0]
   + A[9,2] * Conv_0_aggregated_features[2,0] ] = 0
 Conv_0_averages_constraint[3,0]: [
   A[0,3] * Conv_0_aggregated_features[3,0]
   + A[1,3] * Conv_0_aggregated_features[3,0]
   + A[2,3] * Conv_0_aggregated_features[3,0] - A[3,0] * X[0,0]
   - A[3,1] * X[1,0] - A[3,2] * X[2,0] - A[3,3] * X[3,0]
   + A[3,3] * Conv_0_aggregated_features[3,0] - A[3,4] * X[4,0]
   - A[3,5] * X[5,0] - A[3,6] * X[6,0] - A[3,7] * X[7,0] - A[3,8] * X[8,0]
   - A[3,9] * X[9,0] + A[4,3] * Conv_0_aggregated_features[3,0]
   + A[5,3] * Conv_0_aggregated_features[3,0]
   + A[6,3] * Conv_0_aggregated_features[3,0]
   + A[7,3] * Conv_0_aggregated_features[3,0]
   + A[8,3] * Conv_0_aggregated_features[3,0]
   + A[9,3] * Conv_0_aggregated_features[3,0] ] = 0
 Conv_0_averages_constraint[4,0]: [
   A[0,4] * Conv_0_aggregated_features[4,0]
   + A[1,4] * Conv_0_aggregated_features[4,0]
   + A[2,4] * Conv_0_aggregated_features[4,0]
   + A[3,4] * Conv_0_aggregated_features[4,0] - A[4,0] * X[0,0]
   - A[4,1] * X[1,0] - A[4,2] * X[2,0] - A[4,3] * X[3,0] - A[4,4] * X[4,0]
   + A[4,4] * Conv_0_aggregated_features[4,0] - A[4,5] * X[5,0]
   - A[4,6] * X[6,0] - A[4,7] * X[7,0] - A[4,8] * X[8,0] - A[4,9] * X[9,0]
   + A[5,4] * Conv_0_aggregated_features[4,0]
   + A[6,4] * Conv_0_aggregated_features[4,0]
   + A[7,4] * Conv_0_aggregated_features[4,0]
   + A[8,4] * Conv_0_aggregated_features[4,0]
   + A[9,4] * Conv_0_aggregated_features[4,0] ] = 0
 Conv_0_averages_constraint[5,0]: [
   A[0,5] * Conv_0_aggregated_features[5,0]
   + A[1,5] * Conv_0_aggregated_features[5,0]
   + A[2,5] * Conv_0_aggregated_features[5,0]
   + A[3,5] * Conv_0_aggregated_features[5,0]
   + A[4,5] * Conv_0_aggregated_features[5,0] - A[5,0] * X[0,0]
   - A[5,1] * X[1,0] - A[5,2] * X[2,0] - A[5,3] * X[3,0] - A[5,4] * X[4,0]
   - A[5,5] * X[5,0] + A[5,5] * Conv_0_aggregated_features[5,0]
   - A[5,6] * X[6,0] - A[5,7] * X[7,0] - A[5,8] * X[8,0] - A[5,9] * X[9,0]
   + A[6,5] * Conv_0_aggregated_features[5,0]
   + A[7,5] * Conv_0_aggregated_features[5,0]
   + A[8,5] * Conv_0_aggregated_features[5,0]
   + A[9,5] * Conv_0_aggregated_features[5,0] ] = 0
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
   - A[6,8] * X[8,0] - A[6,9] * X[9,0]
   + A[7,6] * Conv_0_aggregated_features[6,0]
   + A[8,6] * Conv_0_aggregated_features[6,0]
   + A[9,6] * Conv_0_aggregated_features[6,0] ] = 0
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
   + A[7,7] * Conv_0_aggregated_features[7,0] - A[7,8] * X[8,0]
   - A[7,9] * X[9,0] + A[8,7] * Conv_0_aggregated_features[7,0]
   + A[9,7] * Conv_0_aggregated_features[7,0] ] = 0
 Conv_0_averages_constraint[8,0]: [
   A[0,8] * Conv_0_aggregated_features[8,0]
   + A[1,8] * Conv_0_aggregated_features[8,0]
   + A[2,8] * Conv_0_aggregated_features[8,0]
   + A[3,8] * Conv_0_aggregated_features[8,0]
   + A[4,8] * Conv_0_aggregated_features[8,0]
   + A[5,8] * Conv_0_aggregated_features[8,0]
   + A[6,8] * Conv_0_aggregated_features[8,0]
   + A[7,8] * Conv_0_aggregated_features[8,0] - A[8,0] * X[0,0]
   - A[8,1] * X[1,0] - A[8,2] * X[2,0] - A[8,3] * X[3,0] - A[8,4] * X[4,0]
   - A[8,5] * X[5,0] - A[8,6] * X[6,0] - A[8,7] * X[7,0] - A[8,8] * X[8,0]
   + A[8,8] * Conv_0_aggregated_features[8,0] - A[8,9] * X[9,0]
   + A[9,8] * Conv_0_aggregated_features[8,0] ] = 0
 Conv_0_averages_constraint[9,0]: [
   A[0,9] * Conv_0_aggregated_features[9,0]
   + A[1,9] * Conv_0_aggregated_features[9,0]
   + A[2,9] * Conv_0_aggregated_features[9,0]
   + A[3,9] * Conv_0_aggregated_features[9,0]
   + A[4,9] * Conv_0_aggregated_features[9,0]
   + A[5,9] * Conv_0_aggregated_features[9,0]
   + A[6,9] * Conv_0_aggregated_features[9,0]
   + A[7,9] * Conv_0_aggregated_features[9,0]
   + A[8,9] * Conv_0_aggregated_features[9,0] - A[9,0] * X[0,0]
   - A[9,1] * X[1,0] - A[9,2] * X[2,0] - A[9,3] * X[3,0] - A[9,4] * X[4,0]
   - A[9,5] * X[5,0] - A[9,6] * X[6,0] - A[9,7] * X[7,0] - A[9,8] * X[8,0]
   - A[9,9] * X[9,0] + A[9,9] * Conv_0_aggregated_features[9,0] ] = 0
 Conv_1_averages_constraint[0,0]: [ - A[0,0] * Conv_0_Relu_ts[0,0]
   + A[0,0] * Conv_1_aggregated_features[0,0]
   - A[0,1] * Conv_0_Relu_ts[1,0] - A[0,2] * Conv_0_Relu_ts[2,0]
   - A[0,3] * Conv_0_Relu_ts[3,0] - A[0,4] * Conv_0_Relu_ts[4,0]
   - A[0,5] * Conv_0_Relu_ts[5,0] - A[0,6] * Conv_0_Relu_ts[6,0]
   - A[0,7] * Conv_0_Relu_ts[7,0] - A[0,8] * Conv_0_Relu_ts[8,0]
   - A[0,9] * Conv_0_Relu_ts[9,0]
   + A[1,0] * Conv_1_aggregated_features[0,0]
   + A[2,0] * Conv_1_aggregated_features[0,0]
   + A[3,0] * Conv_1_aggregated_features[0,0]
   + A[4,0] * Conv_1_aggregated_features[0,0]
   + A[5,0] * Conv_1_aggregated_features[0,0]
   + A[6,0] * Conv_1_aggregated_features[0,0]
   + A[7,0] * Conv_1_aggregated_features[0,0]
   + A[8,0] * Conv_1_aggregated_features[0,0]
   + A[9,0] * Conv_1_aggregated_features[0,0] ] = 0
 Conv_1_averages_constraint[0,1]: [ - A[0,0] * Conv_0_Relu_ts[0,1]
   + A[0,0] * Conv_1_aggregated_features[0,1]
   - A[0,1] * Conv_0_Relu_ts[1,1] - A[0,2] * Conv_0_Relu_ts[2,1]
   - A[0,3] * Conv_0_Relu_ts[3,1] - A[0,4] * Conv_0_Relu_ts[4,1]
   - A[0,5] * Conv_0_Relu_ts[5,1] - A[0,6] * Conv_0_Relu_ts[6,1]
   - A[0,7] * Conv_0_Relu_ts[7,1] - A[0,8] * Conv_0_Relu_ts[8,1]
   - A[0,9] * Conv_0_Relu_ts[9,1]
   + A[1,0] * Conv_1_aggregated_features[0,1]
   + A[2,0] * Conv_1_aggregated_features[0,1]
   + A[3,0] * Conv_1_aggregated_features[0,1]
   + A[4,0] * Conv_1_aggregated_features[0,1]
   + A[5,0] * Conv_1_aggregated_features[0,1]
   + A[6,0] * Conv_1_aggregated_features[0,1]
   + A[7,0] * Conv_1_aggregated_features[0,1]
   + A[8,0] * Conv_1_aggregated_features[0,1]
   + A[9,0] * Conv_1_aggregated_features[0,1] ] = 0
 Conv_1_averages_constraint[0,2]: [ - A[0,0] * Conv_0_Relu_ts[0,2]
   + A[0,0] * Conv_1_aggregated_features[0,2]
   - A[0,1] * Conv_0_Relu_ts[1,2] - A[0,2] * Conv_0_Relu_ts[2,2]
   - A[0,3] * Conv_0_Relu_ts[3,2] - A[0,4] * Conv_0_Relu_ts[4,2]
   - A[0,5] * Conv_0_Relu_ts[5,2] - A[0,6] * Conv_0_Relu_ts[6,2]
   - A[0,7] * Conv_0_Relu_ts[7,2] - A[0,8] * Conv_0_Relu_ts[8,2]
   - A[0,9] * Conv_0_Relu_ts[9,2]
   + A[1,0] * Conv_1_aggregated_features[0,2]
   + A[2,0] * Conv_1_aggregated_features[0,2]
   + A[3,0] * Conv_1_aggregated_features[0,2]
   + A[4,0] * Conv_1_aggregated_features[0,2]
   + A[5,0] * Conv_1_aggregated_features[0,2]
   + A[6,0] * Conv_1_aggregated_features[0,2]
   + A[7,0] * Conv_1_aggregated_features[0,2]
   + A[8,0] * Conv_1_aggregated_features[0,2]
   + A[9,0] * Conv_1_aggregated_features[0,2] ] = 0
 Conv_1_averages_constraint[0,3]: [ - A[0,0] * Conv_0_Relu_ts[0,3]
   + A[0,0] * Conv_1_aggregated_features[0,3]
   - A[0,1] * Conv_0_Relu_ts[1,3] - A[0,2] * Conv_0_Relu_ts[2,3]
   - A[0,3] * Conv_0_Relu_ts[3,3] - A[0,4] * Conv_0_Relu_ts[4,3]
   - A[0,5] * Conv_0_Relu_ts[5,3] - A[0,6] * Conv_0_Relu_ts[6,3]
   - A[0,7] * Conv_0_Relu_ts[7,3] - A[0,8] * Conv_0_Relu_ts[8,3]
   - A[0,9] * Conv_0_Relu_ts[9,3]
   + A[1,0] * Conv_1_aggregated_features[0,3]
   + A[2,0] * Conv_1_aggregated_features[0,3]
   + A[3,0] * Conv_1_aggregated_features[0,3]
   + A[4,0] * Conv_1_aggregated_features[0,3]
   + A[5,0] * Conv_1_aggregated_features[0,3]
   + A[6,0] * Conv_1_aggregated_features[0,3]
   + A[7,0] * Conv_1_aggregated_features[0,3]
   + A[8,0] * Conv_1_aggregated_features[0,3]
   + A[9,0] * Conv_1_aggregated_features[0,3] ] = 0
 Conv_1_averages_constraint[0,4]: [ - A[0,0] * Conv_0_Relu_ts[0,4]
   + A[0,0] * Conv_1_aggregated_features[0,4]
   - A[0,1] * Conv_0_Relu_ts[1,4] - A[0,2] * Conv_0_Relu_ts[2,4]
   - A[0,3] * Conv_0_Relu_ts[3,4] - A[0,4] * Conv_0_Relu_ts[4,4]
   - A[0,5] * Conv_0_Relu_ts[5,4] - A[0,6] * Conv_0_Relu_ts[6,4]
   - A[0,7] * Conv_0_Relu_ts[7,4] - A[0,8] * Conv_0_Relu_ts[8,4]
   - A[0,9] * Conv_0_Relu_ts[9,4]
   + A[1,0] * Conv_1_aggregated_features[0,4]
   + A[2,0] * Conv_1_aggregated_features[0,4]
   + A[3,0] * Conv_1_aggregated_features[0,4]
   + A[4,0] * Conv_1_aggregated_features[0,4]
   + A[5,0] * Conv_1_aggregated_features[0,4]
   + A[6,0] * Conv_1_aggregated_features[0,4]
   + A[7,0] * Conv_1_aggregated_features[0,4]
   + A[8,0] * Conv_1_aggregated_features[0,4]
   + A[9,0] * Conv_1_aggregated_features[0,4] ] = 0
 Conv_1_averages_constraint[0,5]: [ - A[0,0] * Conv_0_Relu_ts[0,5]
   + A[0,0] * Conv_1_aggregated_features[0,5]
   - A[0,1] * Conv_0_Relu_ts[1,5] - A[0,2] * Conv_0_Relu_ts[2,5]
   - A[0,3] * Conv_0_Relu_ts[3,5] - A[0,4] * Conv_0_Relu_ts[4,5]
   - A[0,5] * Conv_0_Relu_ts[5,5] - A[0,6] * Conv_0_Relu_ts[6,5]
   - A[0,7] * Conv_0_Relu_ts[7,5] - A[0,8] * Conv_0_Relu_ts[8,5]
   - A[0,9] * Conv_0_Relu_ts[9,5]
   + A[1,0] * Conv_1_aggregated_features[0,5]
   + A[2,0] * Conv_1_aggregated_features[0,5]
   + A[3,0] * Conv_1_aggregated_features[0,5]
   + A[4,0] * Conv_1_aggregated_features[0,5]
   + A[5,0] * Conv_1_aggregated_features[0,5]
   + A[6,0] * Conv_1_aggregated_features[0,5]
   + A[7,0] * Conv_1_aggregated_features[0,5]
   + A[8,0] * Conv_1_aggregated_features[0,5]
   + A[9,0] * Conv_1_aggregated_features[0,5] ] = 0
 Conv_1_averages_constraint[0,6]: [ - A[0,0] * Conv_0_Relu_ts[0,6]
   + A[0,0] * Conv_1_aggregated_features[0,6]
   - A[0,1] * Conv_0_Relu_ts[1,6] - A[0,2] * Conv_0_Relu_ts[2,6]
   - A[0,3] * Conv_0_Relu_ts[3,6] - A[0,4] * Conv_0_Relu_ts[4,6]
   - A[0,5] * Conv_0_Relu_ts[5,6] - A[0,6] * Conv_0_Relu_ts[6,6]
   - A[0,7] * Conv_0_Relu_ts[7,6] - A[0,8] * Conv_0_Relu_ts[8,6]
   - A[0,9] * Conv_0_Relu_ts[9,6]
   + A[1,0] * Conv_1_aggregated_features[0,6]
   + A[2,0] * Conv_1_aggregated_features[0,6]
   + A[3,0] * Conv_1_aggregated_features[0,6]
   + A[4,0] * Conv_1_aggregated_features[0,6]
   + A[5,0] * Conv_1_aggregated_features[0,6]
   + A[6,0] * Conv_1_aggregated_features[0,6]
   + A[7,0] * Conv_1_aggregated_features[0,6]
   + A[8,0] * Conv_1_aggregated_features[0,6]
   + A[9,0] * Conv_1_aggregated_features[0,6] ] = 0
 Conv_1_averages_constraint[0,7]: [ - A[0,0] * Conv_0_Relu_ts[0,7]
   + A[0,0] * Conv_1_aggregated_features[0,7]
   - A[0,1] * Conv_0_Relu_ts[1,7] - A[0,2] * Conv_0_Relu_ts[2,7]
   - A[0,3] * Conv_0_Relu_ts[3,7] - A[0,4] * Conv_0_Relu_ts[4,7]
   - A[0,5] * Conv_0_Relu_ts[5,7] - A[0,6] * Conv_0_Relu_ts[6,7]
   - A[0,7] * Conv_0_Relu_ts[7,7] - A[0,8] * Conv_0_Relu_ts[8,7]
   - A[0,9] * Conv_0_Relu_ts[9,7]
   + A[1,0] * Conv_1_aggregated_features[0,7]
   + A[2,0] * Conv_1_aggregated_features[0,7]
   + A[3,0] * Conv_1_aggregated_features[0,7]
   + A[4,0] * Conv_1_aggregated_features[0,7]
   + A[5,0] * Conv_1_aggregated_features[0,7]
   + A[6,0] * Conv_1_aggregated_features[0,7]
   + A[7,0] * Conv_1_aggregated_features[0,7]
   + A[8,0] * Conv_1_aggregated_features[0,7]
   + A[9,0] * Conv_1_aggregated_features[0,7] ] = 0
 Conv_1_averages_constraint[0,8]: [ - A[0,0] * Conv_0_Relu_ts[0,8]
   + A[0,0] * Conv_1_aggregated_features[0,8]
   - A[0,1] * Conv_0_Relu_ts[1,8] - A[0,2] * Conv_0_Relu_ts[2,8]
   - A[0,3] * Conv_0_Relu_ts[3,8] - A[0,4] * Conv_0_Relu_ts[4,8]
   - A[0,5] * Conv_0_Relu_ts[5,8] - A[0,6] * Conv_0_Relu_ts[6,8]
   - A[0,7] * Conv_0_Relu_ts[7,8] - A[0,8] * Conv_0_Relu_ts[8,8]
   - A[0,9] * Conv_0_Relu_ts[9,8]
   + A[1,0] * Conv_1_aggregated_features[0,8]
   + A[2,0] * Conv_1_aggregated_features[0,8]
   + A[3,0] * Conv_1_aggregated_features[0,8]
   + A[4,0] * Conv_1_aggregated_features[0,8]
   + A[5,0] * Conv_1_aggregated_features[0,8]
   + A[6,0] * Conv_1_aggregated_features[0,8]
   + A[7,0] * Conv_1_aggregated_features[0,8]
   + A[8,0] * Conv_1_aggregated_features[0,8]
   + A[9,0] * Conv_1_aggregated_features[0,8] ] = 0
 Conv_1_averages_constraint[0,9]: [ - A[0,0] * Conv_0_Relu_ts[0,9]
   + A[0,0] * Conv_1_aggregated_features[0,9]
   - A[0,1] * Conv_0_Relu_ts[1,9] - A[0,2] * Conv_0_Relu_ts[2,9]
   - A[0,3] * Conv_0_Relu_ts[3,9] - A[0,4] * Conv_0_Relu_ts[4,9]
   - A[0,5] * Conv_0_Relu_ts[5,9] - A[0,6] * Conv_0_Relu_ts[6,9]
   - A[0,7] * Conv_0_Relu_ts[7,9] - A[0,8] * Conv_0_Relu_ts[8,9]
   - A[0,9] * Conv_0_Relu_ts[9,9]
   + A[1,0] * Conv_1_aggregated_features[0,9]
   + A[2,0] * Conv_1_aggregated_features[0,9]
   + A[3,0] * Conv_1_aggregated_features[0,9]
   + A[4,0] * Conv_1_aggregated_features[0,9]
   + A[5,0] * Conv_1_aggregated_features[0,9]
   + A[6,0] * Conv_1_aggregated_features[0,9]
   + A[7,0] * Conv_1_aggregated_features[0,9]
   + A[8,0] * Conv_1_aggregated_features[0,9]
   + A[9,0] * Conv_1_aggregated_features[0,9] ] = 0
 Conv_1_averages_constraint[0,10]: [ - A[0,0] * Conv_0_Relu_ts[0,10]
   + A[0,0] * Conv_1_aggregated_features[0,10]
   - A[0,1] * Conv_0_Relu_ts[1,10] - A[0,2] * Conv_0_Relu_ts[2,10]
   - A[0,3] * Conv_0_Relu_ts[3,10] - A[0,4] * Conv_0_Relu_ts[4,10]
   - A[0,5] * Conv_0_Relu_ts[5,10] - A[0,6] * Conv_0_Relu_ts[6,10]
   - A[0,7] * Conv_0_Relu_ts[7,10] - A[0,8] * Conv_0_Relu_ts[8,10]
   - A[0,9] * Conv_0_Relu_ts[9,10]
   + A[1,0] * Conv_1_aggregated_features[0,10]
   + A[2,0] * Conv_1_aggregated_features[0,10]
   + A[3,0] * Conv_1_aggregated_features[0,10]
   + A[4,0] * Conv_1_aggregated_features[0,10]
   + A[5,0] * Conv_1_aggregated_features[0,10]
   + A[6,0] * Conv_1_aggregated_features[0,10]
   + A[7,0] * Conv_1_aggregated_features[0,10]
   + A[8,0] * Conv_1_aggregated_features[0,10]
   + A[9,0] * Conv_1_aggregated_features[0,10] ] = 0
 Conv_1_averages_constraint[0,11]: [ - A[0,0] * Conv_0_Relu_ts[0,11]
   + A[0,0] * Conv_1_aggregated_features[0,11]
   - A[0,1] * Conv_0_Relu_ts[1,11] - A[0,2] * Conv_0_Relu_ts[2,11]
   - A[0,3] * Conv_0_Relu_ts[3,11] - A[0,4] * Conv_0_Relu_ts[4,11]
   - A[0,5] * Conv_0_Relu_ts[5,11] - A[0,6] * Conv_0_Relu_ts[6,11]
   - A[0,7] * Conv_0_Relu_ts[7,11] - A[0,8] * Conv_0_Relu_ts[8,11]
   - A[0,9] * Conv_0_Relu_ts[9,11]
   + A[1,0] * Conv_1_aggregated_features[0,11]
   + A[2,0] * Conv_1_aggregated_features[0,11]
   + A[3,0] * Conv_1_aggregated_features[0,11]
   + A[4,0] * Conv_1_aggregated_features[0,11]
   + A[5,0] * Conv_1_aggregated_features[0,11]
   + A[6,0] * Conv_1_aggregated_features[0,11]
   + A[7,0] * Conv_1_aggregated_features[0,11]
   + A[8,0] * Conv_1_aggregated_features[0,11]
   + A[9,0] * Conv_1_aggregated_features[0,11] ] = 0
 Conv_1_averages_constraint[0,12]: [ - A[0,0] * Conv_0_Relu_ts[0,12]
   + A[0,0] * Conv_1_aggregated_features[0,12]
   - A[0,1] * Conv_0_Relu_ts[1,12] - A[0,2] * Conv_0_Relu_ts[2,12]
   - A[0,3] * Conv_0_Relu_ts[3,12] - A[0,4] * Conv_0_Relu_ts[4,12]
   - A[0,5] * Conv_0_Relu_ts[5,12] - A[0,6] * Conv_0_Relu_ts[6,12]
   - A[0,7] * Conv_0_Relu_ts[7,12] - A[0,8] * Conv_0_Relu_ts[8,12]
   - A[0,9] * Conv_0_Relu_ts[9,12]
   + A[1,0] * Conv_1_aggregated_features[0,12]
   + A[2,0] * Conv_1_aggregated_features[0,12]
   + A[3,0] * Conv_1_aggregated_features[0,12]
   + A[4,0] * Conv_1_aggregated_features[0,12]
   + A[5,0] * Conv_1_aggregated_features[0,12]
   + A[6,0] * Conv_1_aggregated_features[0,12]
   + A[7,0] * Conv_1_aggregated_features[0,12]
   + A[8,0] * Conv_1_aggregated_features[0,12]
   + A[9,0] * Conv_1_aggregated_features[0,12] ] = 0
 Conv_1_averages_constraint[0,13]: [ - A[0,0] * Conv_0_Relu_ts[0,13]
   + A[0,0] * Conv_1_aggregated_features[0,13]
   - A[0,1] * Conv_0_Relu_ts[1,13] - A[0,2] * Conv_0_Relu_ts[2,13]
   - A[0,3] * Conv_0_Relu_ts[3,13] - A[0,4] * Conv_0_Relu_ts[4,13]
   - A[0,5] * Conv_0_Relu_ts[5,13] - A[0,6] * Conv_0_Relu_ts[6,13]
   - A[0,7] * Conv_0_Relu_ts[7,13] - A[0,8] * Conv_0_Relu_ts[8,13]
   - A[0,9] * Conv_0_Relu_ts[9,13]
   + A[1,0] * Conv_1_aggregated_features[0,13]
   + A[2,0] * Conv_1_aggregated_features[0,13]
   + A[3,0] * Conv_1_aggregated_features[0,13]
   + A[4,0] * Conv_1_aggregated_features[0,13]
   + A[5,0] * Conv_1_aggregated_features[0,13]
   + A[6,0] * Conv_1_aggregated_features[0,13]
   + A[7,0] * Conv_1_aggregated_features[0,13]
   + A[8,0] * Conv_1_aggregated_features[0,13]
   + A[9,0] * Conv_1_aggregated_features[0,13] ] = 0
 Conv_1_averages_constraint[0,14]: [ - A[0,0] * Conv_0_Relu_ts[0,14]
   + A[0,0] * Conv_1_aggregated_features[0,14]
   - A[0,1] * Conv_0_Relu_ts[1,14] - A[0,2] * Conv_0_Relu_ts[2,14]
   - A[0,3] * Conv_0_Relu_ts[3,14] - A[0,4] * Conv_0_Relu_ts[4,14]
   - A[0,5] * Conv_0_Relu_ts[5,14] - A[0,6] * Conv_0_Relu_ts[6,14]
   - A[0,7] * Conv_0_Relu_ts[7,14] - A[0,8] * Conv_0_Relu_ts[8,14]
   - A[0,9] * Conv_0_Relu_ts[9,14]
   + A[1,0] * Conv_1_aggregated_features[0,14]
   + A[2,0] * Conv_1_aggregated_features[0,14]
   + A[3,0] * Conv_1_aggregated_features[0,14]
   + A[4,0] * Conv_1_aggregated_features[0,14]
   + A[5,0] * Conv_1_aggregated_features[0,14]
   + A[6,0] * Conv_1_aggregated_features[0,14]
   + A[7,0] * Conv_1_aggregated_features[0,14]
   + A[8,0] * Conv_1_aggregated_features[0,14]
   + A[9,0] * Conv_1_aggregated_features[0,14] ] = 0
 Conv_1_averages_constraint[0,15]: [ - A[0,0] * Conv_0_Relu_ts[0,15]
   + A[0,0] * Conv_1_aggregated_features[0,15]
   - A[0,1] * Conv_0_Relu_ts[1,15] - A[0,2] * Conv_0_Relu_ts[2,15]
   - A[0,3] * Conv_0_Relu_ts[3,15] - A[0,4] * Conv_0_Relu_ts[4,15]
   - A[0,5] * Conv_0_Relu_ts[5,15] - A[0,6] * Conv_0_Relu_ts[6,15]
   - A[0,7] * Conv_0_Relu_ts[7,15] - A[0,8] * Conv_0_Relu_ts[8,15]
   - A[0,9] * Conv_0_Relu_ts[9,15]
   + A[1,0] * Conv_1_aggregated_features[0,15]
   + A[2,0] * Conv_1_aggregated_features[0,15]
   + A[3,0] * Conv_1_aggregated_features[0,15]
   + A[4,0] * Conv_1_aggregated_features[0,15]
   + A[5,0] * Conv_1_aggregated_features[0,15]
   + A[6,0] * Conv_1_aggregated_features[0,15]
   + A[7,0] * Conv_1_aggregated_features[0,15]
   + A[8,0] * Conv_1_aggregated_features[0,15]
   + A[9,0] * Conv_1_aggregated_features[0,15] ] = 0
 Conv_1_averages_constraint[1,0]: [
   A[0,1] * Conv_1_aggregated_features[1,0] - A[1,0] * Conv_0_Relu_ts[0,0]
   - A[1,1] * Conv_0_Relu_ts[1,0]
   + A[1,1] * Conv_1_aggregated_features[1,0]
   - A[1,2] * Conv_0_Relu_ts[2,0] - A[1,3] * Conv_0_Relu_ts[3,0]
   - A[1,4] * Conv_0_Relu_ts[4,0] - A[1,5] * Conv_0_Relu_ts[5,0]
   - A[1,6] * Conv_0_Relu_ts[6,0] - A[1,7] * Conv_0_Relu_ts[7,0]
   - A[1,8] * Conv_0_Relu_ts[8,0] - A[1,9] * Conv_0_Relu_ts[9,0]
   + A[2,1] * Conv_1_aggregated_features[1,0]
   + A[3,1] * Conv_1_aggregated_features[1,0]
   + A[4,1] * Conv_1_aggregated_features[1,0]
   + A[5,1] * Conv_1_aggregated_features[1,0]
   + A[6,1] * Conv_1_aggregated_features[1,0]
   + A[7,1] * Conv_1_aggregated_features[1,0]
   + A[8,1] * Conv_1_aggregated_features[1,0]
   + A[9,1] * Conv_1_aggregated_features[1,0] ] = 0
 Conv_1_averages_constraint[1,1]: [
   A[0,1] * Conv_1_aggregated_features[1,1] - A[1,0] * Conv_0_Relu_ts[0,1]
   - A[1,1] * Conv_0_Relu_ts[1,1]
   + A[1,1] * Conv_1_aggregated_features[1,1]
   - A[1,2] * Conv_0_Relu_ts[2,1] - A[1,3] * Conv_0_Relu_ts[3,1]
   - A[1,4] * Conv_0_Relu_ts[4,1] - A[1,5] * Conv_0_Relu_ts[5,1]
   - A[1,6] * Conv_0_Relu_ts[6,1] - A[1,7] * Conv_0_Relu_ts[7,1]
   - A[1,8] * Conv_0_Relu_ts[8,1] - A[1,9] * Conv_0_Relu_ts[9,1]
   + A[2,1] * Conv_1_aggregated_features[1,1]
   + A[3,1] * Conv_1_aggregated_features[1,1]
   + A[4,1] * Conv_1_aggregated_features[1,1]
   + A[5,1] * Conv_1_aggregated_features[1,1]
   + A[6,1] * Conv_1_aggregated_features[1,1]
   + A[7,1] * Conv_1_aggregated_features[1,1]
   + A[8,1] * Conv_1_aggregated_features[1,1]
   + A[9,1] * Conv_1_aggregated_features[1,1] ] = 0
 Conv_1_averages_constraint[1,2]: [
   A[0,1] * Conv_1_aggregated_features[1,2] - A[1,0] * Conv_0_Relu_ts[0,2]
   - A[1,1] * Conv_0_Relu_ts[1,2]
   + A[1,1] * Conv_1_aggregated_features[1,2]
   - A[1,2] * Conv_0_Relu_ts[2,2] - A[1,3] * Conv_0_Relu_ts[3,2]
   - A[1,4] * Conv_0_Relu_ts[4,2] - A[1,5] * Conv_0_Relu_ts[5,2]
   - A[1,6] * Conv_0_Relu_ts[6,2] - A[1,7] * Conv_0_Relu_ts[7,2]
   - A[1,8] * Conv_0_Relu_ts[8,2] - A[1,9] * Conv_0_Relu_ts[9,2]
   + A[2,1] * Conv_1_aggregated_features[1,2]
   + A[3,1] * Conv_1_aggregated_features[1,2]
   + A[4,1] * Conv_1_aggregated_features[1,2]
   + A[5,1] * Conv_1_aggregated_features[1,2]
   + A[6,1] * Conv_1_aggregated_features[1,2]
   + A[7,1] * Conv_1_aggregated_features[1,2]
   + A[8,1] * Conv_1_aggregated_features[1,2]
   + A[9,1] * Conv_1_aggregated_features[1,2] ] = 0
 Conv_1_averages_constraint[1,3]: [
   A[0,1] * Conv_1_aggregated_features[1,3] - A[1,0] * Conv_0_Relu_ts[0,3]
   - A[1,1] * Conv_0_Relu_ts[1,3]
   + A[1,1] * Conv_1_aggregated_features[1,3]
   - A[1,2] * Conv_0_Relu_ts[2,3] - A[1,3] * Conv_0_Relu_ts[3,3]
   - A[1,4] * Conv_0_Relu_ts[4,3] - A[1,5] * Conv_0_Relu_ts[5,3]
   - A[1,6] * Conv_0_Relu_ts[6,3] - A[1,7] * Conv_0_Relu_ts[7,3]
   - A[1,8] * Conv_0_Relu_ts[8,3] - A[1,9] * Conv_0_Relu_ts[9,3]
   + A[2,1] * Conv_1_aggregated_features[1,3]
   + A[3,1] * Conv_1_aggregated_features[1,3]
   + A[4,1] * Conv_1_aggregated_features[1,3]
   + A[5,1] * Conv_1_aggregated_features[1,3]
   + A[6,1] * Conv_1_aggregated_features[1,3]
   + A[7,1] * Conv_1_aggregated_features[1,3]
   + A[8,1] * Conv_1_aggregated_features[1,3]
   + A[9,1] * Conv_1_aggregated_features[1,3] ] = 0
 Conv_1_averages_constraint[1,4]: [
   A[0,1] * Conv_1_aggregated_features[1,4] - A[1,0] * Conv_0_Relu_ts[0,4]
   - A[1,1] * Conv_0_Relu_ts[1,4]
   + A[1,1] * Conv_1_aggregated_features[1,4]
   - A[1,2] * Conv_0_Relu_ts[2,4] - A[1,3] * Conv_0_Relu_ts[3,4]
   - A[1,4] * Conv_0_Relu_ts[4,4] - A[1,5] * Conv_0_Relu_ts[5,4]
   - A[1,6] * Conv_0_Relu_ts[6,4] - A[1,7] * Conv_0_Relu_ts[7,4]
   - A[1,8] * Conv_0_Relu_ts[8,4] - A[1,9] * Conv_0_Relu_ts[9,4]
   + A[2,1] * Conv_1_aggregated_features[1,4]
   + A[3,1] * Conv_1_aggregated_features[1,4]
   + A[4,1] * Conv_1_aggregated_features[1,4]
   + A[5,1] * Conv_1_aggregated_features[1,4]
   + A[6,1] * Conv_1_aggregated_features[1,4]
   + A[7,1] * Conv_1_aggregated_features[1,4]
   + A[8,1] * Conv_1_aggregated_features[1,4]
   + A[9,1] * Conv_1_aggregated_features[1,4] ] = 0
 Conv_1_averages_constraint[1,5]: [
   A[0,1] * Conv_1_aggregated_features[1,5] - A[1,0] * Conv_0_Relu_ts[0,5]
   - A[1,1] * Conv_0_Relu_ts[1,5]
   + A[1,1] * Conv_1_aggregated_features[1,5]
   - A[1,2] * Conv_0_Relu_ts[2,5] - A[1,3] * Conv_0_Relu_ts[3,5]
   - A[1,4] * Conv_0_Relu_ts[4,5] - A[1,5] * Conv_0_Relu_ts[5,5]
   - A[1,6] * Conv_0_Relu_ts[6,5] - A[1,7] * Conv_0_Relu_ts[7,5]
   - A[1,8] * Conv_0_Relu_ts[8,5] - A[1,9] * Conv_0_Relu_ts[9,5]
   + A[2,1] * Conv_1_aggregated_features[1,5]
   + A[3,1] * Conv_1_aggregated_features[1,5]
   + A[4,1] * Conv_1_aggregated_features[1,5]
   + A[5,1] * Conv_1_aggregated_features[1,5]
   + A[6,1] * Conv_1_aggregated_features[1,5]
   + A[7,1] * Conv_1_aggregated_features[1,5]
   + A[8,1] * Conv_1_aggregated_features[1,5]
   + A[9,1] * Conv_1_aggregated_features[1,5] ] = 0
 Conv_1_averages_constraint[1,6]: [
   A[0,1] * Conv_1_aggregated_features[1,6] - A[1,0] * Conv_0_Relu_ts[0,6]
   - A[1,1] * Conv_0_Relu_ts[1,6]
   + A[1,1] * Conv_1_aggregated_features[1,6]
   - A[1,2] * Conv_0_Relu_ts[2,6] - A[1,3] * Conv_0_Relu_ts[3,6]
   - A[1,4] * Conv_0_Relu_ts[4,6] - A[1,5] * Conv_0_Relu_ts[5,6]
   - A[1,6] * Conv_0_Relu_ts[6,6] - A[1,7] * Conv_0_Relu_ts[7,6]
   - A[1,8] * Conv_0_Relu_ts[8,6] - A[1,9] * Conv_0_Relu_ts[9,6]
   + A[2,1] * Conv_1_aggregated_features[1,6]
   + A[3,1] * Conv_1_aggregated_features[1,6]
   + A[4,1] * Conv_1_aggregated_features[1,6]
   + A[5,1] * Conv_1_aggregated_features[1,6]
   + A[6,1] * Conv_1_aggregated_features[1,6]
   + A[7,1] * Conv_1_aggregated_features[1,6]
   + A[8,1] * Conv_1_aggregated_features[1,6]
   + A[9,1] * Conv_1_aggregated_features[1,6] ] = 0
 Conv_1_averages_constraint[1,7]: [
   A[0,1] * Conv_1_aggregated_features[1,7] - A[1,0] * Conv_0_Relu_ts[0,7]
   - A[1,1] * Conv_0_Relu_ts[1,7]
   + A[1,1] * Conv_1_aggregated_features[1,7]
   - A[1,2] * Conv_0_Relu_ts[2,7] - A[1,3] * Conv_0_Relu_ts[3,7]
   - A[1,4] * Conv_0_Relu_ts[4,7] - A[1,5] * Conv_0_Relu_ts[5,7]
   - A[1,6] * Conv_0_Relu_ts[6,7] - A[1,7] * Conv_0_Relu_ts[7,7]
   - A[1,8] * Conv_0_Relu_ts[8,7] - A[1,9] * Conv_0_Relu_ts[9,7]
   + A[2,1] * Conv_1_aggregated_features[1,7]
   + A[3,1] * Conv_1_aggregated_features[1,7]
   + A[4,1] * Conv_1_aggregated_features[1,7]
   + A[5,1] * Conv_1_aggregated_features[1,7]
   + A[6,1] * Conv_1_aggregated_features[1,7]
   + A[7,1] * Conv_1_aggregated_features[1,7]
   + A[8,1] * Conv_1_aggregated_features[1,7]
   + A[9,1] * Conv_1_aggregated_features[1,7] ] = 0
 Conv_1_averages_constraint[1,8]: [
   A[0,1] * Conv_1_aggregated_features[1,8] - A[1,0] * Conv_0_Relu_ts[0,8]
   - A[1,1] * Conv_0_Relu_ts[1,8]
   + A[1,1] * Conv_1_aggregated_features[1,8]
   - A[1,2] * Conv_0_Relu_ts[2,8] - A[1,3] * Conv_0_Relu_ts[3,8]
   - A[1,4] * Conv_0_Relu_ts[4,8] - A[1,5] * Conv_0_Relu_ts[5,8]
   - A[1,6] * Conv_0_Relu_ts[6,8] - A[1,7] * Conv_0_Relu_ts[7,8]
   - A[1,8] * Conv_0_Relu_ts[8,8] - A[1,9] * Conv_0_Relu_ts[9,8]
   + A[2,1] * Conv_1_aggregated_features[1,8]
   + A[3,1] * Conv_1_aggregated_features[1,8]
   + A[4,1] * Conv_1_aggregated_features[1,8]
   + A[5,1] * Conv_1_aggregated_features[1,8]
   + A[6,1] * Conv_1_aggregated_features[1,8]
   + A[7,1] * Conv_1_aggregated_features[1,8]
   + A[8,1] * Conv_1_aggregated_features[1,8]
   + A[9,1] * Conv_1_aggregated_features[1,8] ] = 0
 Conv_1_averages_constraint[1,9]: [
   A[0,1] * Conv_1_aggregated_features[1,9] - A[1,0] * Conv_0_Relu_ts[0,9]
   - A[1,1] * Conv_0_Relu_ts[1,9]
   + A[1,1] * Conv_1_aggregated_features[1,9]
   - A[1,2] * Conv_0_Relu_ts[2,9] - A[1,3] * Conv_0_Relu_ts[3,9]
   - A[1,4] * Conv_0_Relu_ts[4,9] - A[1,5] * Conv_0_Relu_ts[5,9]
   - A[1,6] * Conv_0_Relu_ts[6,9] - A[1,7] * Conv_0_Relu_ts[7,9]
   - A[1,8] * Conv_0_Relu_ts[8,9] - A[1,9] * Conv_0_Relu_ts[9,9]
   + A[2,1] * Conv_1_aggregated_features[1,9]
   + A[3,1] * Conv_1_aggregated_features[1,9]
   + A[4,1] * Conv_1_aggregated_features[1,9]
   + A[5,1] * Conv_1_aggregated_features[1,9]
   + A[6,1] * Conv_1_aggregated_features[1,9]
   + A[7,1] * Conv_1_aggregated_features[1,9]
   + A[8,1] * Conv_1_aggregated_features[1,9]
   + A[9,1] * Conv_1_aggregated_features[1,9] ] = 0
 Conv_1_averages_constraint[1,10]: [
   A[0,1] * Conv_1_aggregated_features[1,10]
   - A[1,0] * Conv_0_Relu_ts[0,10] - A[1,1] * Conv_0_Relu_ts[1,10]
   + A[1,1] * Conv_1_aggregated_features[1,10]
   - A[1,2] * Conv_0_Relu_ts[2,10] - A[1,3] * Conv_0_Relu_ts[3,10]
   - A[1,4] * Conv_0_Relu_ts[4,10] - A[1,5] * Conv_0_Relu_ts[5,10]
   - A[1,6] * Conv_0_Relu_ts[6,10] - A[1,7] * Conv_0_Relu_ts[7,10]
   - A[1,8] * Conv_0_Relu_ts[8,10] - A[1,9] * Conv_0_Relu_ts[9,10]
   + A[2,1] * Conv_1_aggregated_features[1,10]
   + A[3,1] * Conv_1_aggregated_features[1,10]
   + A[4,1] * Conv_1_aggregated_features[1,10]
   + A[5,1] * Conv_1_aggregated_features[1,10]
   + A[6,1] * Conv_1_aggregated_features[1,10]
   + A[7,1] * Conv_1_aggregated_features[1,10]
   + A[8,1] * Conv_1_aggregated_features[1,10]
   + A[9,1] * Conv_1_aggregated_features[1,10] ] = 0
 Conv_1_averages_constraint[1,11]: [
   A[0,1] * Conv_1_aggregated_features[1,11]
   - A[1,0] * Conv_0_Relu_ts[0,11] - A[1,1] * Conv_0_Relu_ts[1,11]
   + A[1,1] * Conv_1_aggregated_features[1,11]
   - A[1,2] * Conv_0_Relu_ts[2,11] - A[1,3] * Conv_0_Relu_ts[3,11]
   - A[1,4] * Conv_0_Relu_ts[4,11] - A[1,5] * Conv_0_Relu_ts[5,11]
   - A[1,6] * Conv_0_Relu_ts[6,11] - A[1,7] * Conv_0_Relu_ts[7,11]
   - A[1,8] * Conv_0_Relu_ts[8,11] - A[1,9] * Conv_0_Relu_ts[9,11]
   + A[2,1] * Conv_1_aggregated_features[1,11]
   + A[3,1] * Conv_1_aggregated_features[1,11]
   + A[4,1] * Conv_1_aggregated_features[1,11]
   + A[5,1] * Conv_1_aggregated_features[1,11]
   + A[6,1] * Conv_1_aggregated_features[1,11]
   + A[7,1] * Conv_1_aggregated_features[1,11]
   + A[8,1] * Conv_1_aggregated_features[1,11]
   + A[9,1] * Conv_1_aggregated_features[1,11] ] = 0
 Conv_1_averages_constraint[1,12]: [
   A[0,1] * Conv_1_aggregated_features[1,12]
   - A[1,0] * Conv_0_Relu_ts[0,12] - A[1,1] * Conv_0_Relu_ts[1,12]
   + A[1,1] * Conv_1_aggregated_features[1,12]
   - A[1,2] * Conv_0_Relu_ts[2,12] - A[1,3] * Conv_0_Relu_ts[3,12]
   - A[1,4] * Conv_0_Relu_ts[4,12] - A[1,5] * Conv_0_Relu_ts[5,12]
   - A[1,6] * Conv_0_Relu_ts[6,12] - A[1,7] * Conv_0_Relu_ts[7,12]
   - A[1,8] * Conv_0_Relu_ts[8,12] - A[1,9] * Conv_0_Relu_ts[9,12]
   + A[2,1] * Conv_1_aggregated_features[1,12]
   + A[3,1] * Conv_1_aggregated_features[1,12]
   + A[4,1] * Conv_1_aggregated_features[1,12]
   + A[5,1] * Conv_1_aggregated_features[1,12]
   + A[6,1] * Conv_1_aggregated_features[1,12]
   + A[7,1] * Conv_1_aggregated_features[1,12]
   + A[8,1] * Conv_1_aggregated_features[1,12]
   + A[9,1] * Conv_1_aggregated_features[1,12] ] = 0
 Conv_1_averages_constraint[1,13]: [
   A[0,1] * Conv_1_aggregated_features[1,13]
   - A[1,0] * Conv_0_Relu_ts[0,13] - A[1,1] * Conv_0_Relu_ts[1,13]
   + A[1,1] * Conv_1_aggregated_features[1,13]
   - A[1,2] * Conv_0_Relu_ts[2,13] - A[1,3] * Conv_0_Relu_ts[3,13]
   - A[1,4] * Conv_0_Relu_ts[4,13] - A[1,5] * Conv_0_Relu_ts[5,13]
   - A[1,6] * Conv_0_Relu_ts[6,13] - A[1,7] * Conv_0_Relu_ts[7,13]
   - A[1,8] * Conv_0_Relu_ts[8,13] - A[1,9] * Conv_0_Relu_ts[9,13]
   + A[2,1] * Conv_1_aggregated_features[1,13]
   + A[3,1] * Conv_1_aggregated_features[1,13]
   + A[4,1] * Conv_1_aggregated_features[1,13]
   + A[5,1] * Conv_1_aggregated_features[1,13]
   + A[6,1] * Conv_1_aggregated_features[1,13]
   + A[7,1] * Conv_1_aggregated_features[1,13]
   + A[8,1] * Conv_1_aggregated_features[1,13]
   + A[9,1] * Conv_1_aggregated_features[1,13] ] = 0
 Conv_1_averages_constraint[1,14]: [
   A[0,1] * Conv_1_aggregated_features[1,14]
   - A[1,0] * Conv_0_Relu_ts[0,14] - A[1,1] * Conv_0_Relu_ts[1,14]
   + A[1,1] * Conv_1_aggregated_features[1,14]
   - A[1,2] * Conv_0_Relu_ts[2,14] - A[1,3] * Conv_0_Relu_ts[3,14]
   - A[1,4] * Conv_0_Relu_ts[4,14] - A[1,5] * Conv_0_Relu_ts[5,14]
   - A[1,6] * Conv_0_Relu_ts[6,14] - A[1,7] * Conv_0_Relu_ts[7,14]
   - A[1,8] * Conv_0_Relu_ts[8,14] - A[1,9] * Conv_0_Relu_ts[9,14]
   + A[2,1] * Conv_1_aggregated_features[1,14]
   + A[3,1] * Conv_1_aggregated_features[1,14]
   + A[4,1] * Conv_1_aggregated_features[1,14]
   + A[5,1] * Conv_1_aggregated_features[1,14]
   + A[6,1] * Conv_1_aggregated_features[1,14]
   + A[7,1] * Conv_1_aggregated_features[1,14]
   + A[8,1] * Conv_1_aggregated_features[1,14]
   + A[9,1] * Conv_1_aggregated_features[1,14] ] = 0
 Conv_1_averages_constraint[1,15]: [
   A[0,1] * Conv_1_aggregated_features[1,15]
   - A[1,0] * Conv_0_Relu_ts[0,15] - A[1,1] * Conv_0_Relu_ts[1,15]
   + A[1,1] * Conv_1_aggregated_features[1,15]
   - A[1,2] * Conv_0_Relu_ts[2,15] - A[1,3] * Conv_0_Relu_ts[3,15]
   - A[1,4] * Conv_0_Relu_ts[4,15] - A[1,5] * Conv_0_Relu_ts[5,15]
   - A[1,6] * Conv_0_Relu_ts[6,15] - A[1,7] * Conv_0_Relu_ts[7,15]
   - A[1,8] * Conv_0_Relu_ts[8,15] - A[1,9] * Conv_0_Relu_ts[9,15]
   + A[2,1] * Conv_1_aggregated_features[1,15]
   + A[3,1] * Conv_1_aggregated_features[1,15]
   + A[4,1] * Conv_1_aggregated_features[1,15]
   + A[5,1] * Conv_1_aggregated_features[1,15]
   + A[6,1] * Conv_1_aggregated_features[1,15]
   + A[7,1] * Conv_1_aggregated_features[1,15]
   + A[8,1] * Conv_1_aggregated_features[1,15]
   + A[9,1] * Conv_1_aggregated_features[1,15] ] = 0
 Conv_1_averages_constraint[2,0]: [
   A[0,2] * Conv_1_aggregated_features[2,0]
   + A[1,2] * Conv_1_aggregated_features[2,0]
   - A[2,0] * Conv_0_Relu_ts[0,0] - A[2,1] * Conv_0_Relu_ts[1,0]
   - A[2,2] * Conv_0_Relu_ts[2,0]
   + A[2,2] * Conv_1_aggregated_features[2,0]
   - A[2,3] * Conv_0_Relu_ts[3,0] - A[2,4] * Conv_0_Relu_ts[4,0]
   - A[2,5] * Conv_0_Relu_ts[5,0] - A[2,6] * Conv_0_Relu_ts[6,0]
   - A[2,7] * Conv_0_Relu_ts[7,0] - A[2,8] * Conv_0_Relu_ts[8,0]
   - A[2,9] * Conv_0_Relu_ts[9,0]
   + A[3,2] * Conv_1_aggregated_features[2,0]
   + A[4,2] * Conv_1_aggregated_features[2,0]
   + A[5,2] * Conv_1_aggregated_features[2,0]
   + A[6,2] * Conv_1_aggregated_features[2,0]
   + A[7,2] * Conv_1_aggregated_features[2,0]
   + A[8,2] * Conv_1_aggregated_features[2,0]
   + A[9,2] * Conv_1_aggregated_features[2,0] ] = 0
 Conv_1_averages_constraint[2,1]: [
   A[0,2] * Conv_1_aggregated_features[2,1]
   + A[1,2] * Conv_1_aggregated_features[2,1]
   - A[2,0] * Conv_0_Relu_ts[0,1] - A[2,1] * Conv_0_Relu_ts[1,1]
   - A[2,2] * Conv_0_Relu_ts[2,1]
   + A[2,2] * Conv_1_aggregated_features[2,1]
   - A[2,3] * Conv_0_Relu_ts[3,1] - A[2,4] * Conv_0_Relu_ts[4,1]
   - A[2,5] * Conv_0_Relu_ts[5,1] - A[2,6] * Conv_0_Relu_ts[6,1]
   - A[2,7] * Conv_0_Relu_ts[7,1] - A[2,8] * Conv_0_Relu_ts[8,1]
   - A[2,9] * Conv_0_Relu_ts[9,1]
   + A[3,2] * Conv_1_aggregated_features[2,1]
   + A[4,2] * Conv_1_aggregated_features[2,1]
   + A[5,2] * Conv_1_aggregated_features[2,1]
   + A[6,2] * Conv_1_aggregated_features[2,1]
   + A[7,2] * Conv_1_aggregated_features[2,1]
   + A[8,2] * Conv_1_aggregated_features[2,1]
   + A[9,2] * Conv_1_aggregated_features[2,1] ] = 0
 Conv_1_averages_constraint[2,2]: [
   A[0,2] * Conv_1_aggregated_features[2,2]
   + A[1,2] * Conv_1_aggregated_features[2,2]
   - A[2,0] * Conv_0_Relu_ts[0,2] - A[2,1] * Conv_0_Relu_ts[1,2]
   - A[2,2] * Conv_0_Relu_ts[2,2]
   + A[2,2] * Conv_1_aggregated_features[2,2]
   - A[2,3] * Conv_0_Relu_ts[3,2] - A[2,4] * Conv_0_Relu_ts[4,2]
   - A[2,5] * Conv_0_Relu_ts[5,2] - A[2,6] * Conv_0_Relu_ts[6,2]
   - A[2,7] * Conv_0_Relu_ts[7,2] - A[2,8] * Conv_0_Relu_ts[8,2]
   - A[2,9] * Conv_0_Relu_ts[9,2]
   + A[3,2] * Conv_1_aggregated_features[2,2]
   + A[4,2] * Conv_1_aggregated_features[2,2]
   + A[5,2] * Conv_1_aggregated_features[2,2]
   + A[6,2] * Conv_1_aggregated_features[2,2]
   + A[7,2] * Conv_1_aggregated_features[2,2]
   + A[8,2] * Conv_1_aggregated_features[2,2]
   + A[9,2] * Conv_1_aggregated_features[2,2] ] = 0
 Conv_1_averages_constraint[2,3]: [
   A[0,2] * Conv_1_aggregated_features[2,3]
   + A[1,2] * Conv_1_aggregated_features[2,3]
   - A[2,0] * Conv_0_Relu_ts[0,3] - A[2,1] * Conv_0_Relu_ts[1,3]
   - A[2,2] * Conv_0_Relu_ts[2,3]
   + A[2,2] * Conv_1_aggregated_features[2,3]
   - A[2,3] * Conv_0_Relu_ts[3,3] - A[2,4] * Conv_0_Relu_ts[4,3]
   - A[2,5] * Conv_0_Relu_ts[5,3] - A[2,6] * Conv_0_Relu_ts[6,3]
   - A[2,7] * Conv_0_Relu_ts[7,3] - A[2,8] * Conv_0_Relu_ts[8,3]
   - A[2,9] * Conv_0_Relu_ts[9,3]
   + A[3,2] * Conv_1_aggregated_features[2,3]
   + A[4,2] * Conv_1_aggregated_features[2,3]
   + A[5,2] * Conv_1_aggregated_features[2,3]
   + A[6,2] * Conv_1_aggregated_features[2,3]
   + A[7,2] * Conv_1_aggregated_features[2,3]
   + A[8,2] * Conv_1_aggregated_features[2,3]
   + A[9,2] * Conv_1_aggregated_features[2,3] ] = 0
 Conv_1_averages_constraint[2,4]: [
   A[0,2] * Conv_1_aggregated_features[2,4]
   + A[1,2] * Conv_1_aggregated_features[2,4]
   - A[2,0] * Conv_0_Relu_ts[0,4] - A[2,1] * Conv_0_Relu_ts[1,4]
   - A[2,2] * Conv_0_Relu_ts[2,4]
   + A[2,2] * Conv_1_aggregated_features[2,4]
   - A[2,3] * Conv_0_Relu_ts[3,4] - A[2,4] * Conv_0_Relu_ts[4,4]
   - A[2,5] * Conv_0_Relu_ts[5,4] - A[2,6] * Conv_0_Relu_ts[6,4]
   - A[2,7] * Conv_0_Relu_ts[7,4] - A[2,8] * Conv_0_Relu_ts[8,4]
   - A[2,9] * Conv_0_Relu_ts[9,4]
   + A[3,2] * Conv_1_aggregated_features[2,4]
   + A[4,2] * Conv_1_aggregated_features[2,4]
   + A[5,2] * Conv_1_aggregated_features[2,4]
   + A[6,2] * Conv_1_aggregated_features[2,4]
   + A[7,2] * Conv_1_aggregated_features[2,4]
   + A[8,2] * Conv_1_aggregated_features[2,4]
   + A[9,2] * Conv_1_aggregated_features[2,4] ] = 0
 Conv_1_averages_constraint[2,5]: [
   A[0,2] * Conv_1_aggregated_features[2,5]
   + A[1,2] * Conv_1_aggregated_features[2,5]
   - A[2,0] * Conv_0_Relu_ts[0,5] - A[2,1] * Conv_0_Relu_ts[1,5]
   - A[2,2] * Conv_0_Relu_ts[2,5]
   + A[2,2] * Conv_1_aggregated_features[2,5]
   - A[2,3] * Conv_0_Relu_ts[3,5] - A[2,4] * Conv_0_Relu_ts[4,5]
   - A[2,5] * Conv_0_Relu_ts[5,5] - A[2,6] * Conv_0_Relu_ts[6,5]
   - A[2,7] * Conv_0_Relu_ts[7,5] - A[2,8] * Conv_0_Relu_ts[8,5]
   - A[2,9] * Conv_0_Relu_ts[9,5]
   + A[3,2] * Conv_1_aggregated_features[2,5]
   + A[4,2] * Conv_1_aggregated_features[2,5]
   + A[5,2] * Conv_1_aggregated_features[2,5]
   + A[6,2] * Conv_1_aggregated_features[2,5]
   + A[7,2] * Conv_1_aggregated_features[2,5]
   + A[8,2] * Conv_1_aggregated_features[2,5]
   + A[9,2] * Conv_1_aggregated_features[2,5] ] = 0
 Conv_1_averages_constraint[2,6]: [
   A[0,2] * Conv_1_aggregated_features[2,6]
   + A[1,2] * Conv_1_aggregated_features[2,6]
   - A[2,0] * Conv_0_Relu_ts[0,6] - A[2,1] * Conv_0_Relu_ts[1,6]
   - A[2,2] * Conv_0_Relu_ts[2,6]
   + A[2,2] * Conv_1_aggregated_features[2,6]
   - A[2,3] * Conv_0_Relu_ts[3,6] - A[2,4] * Conv_0_Relu_ts[4,6]
   - A[2,5] * Conv_0_Relu_ts[5,6] - A[2,6] * Conv_0_Relu_ts[6,6]
   - A[2,7] * Conv_0_Relu_ts[7,6] - A[2,8] * Conv_0_Relu_ts[8,6]
   - A[2,9] * Conv_0_Relu_ts[9,6]
   + A[3,2] * Conv_1_aggregated_features[2,6]
   + A[4,2] * Conv_1_aggregated_features[2,6]
   + A[5,2] * Conv_1_aggregated_features[2,6]
   + A[6,2] * Conv_1_aggregated_features[2,6]
   + A[7,2] * Conv_1_aggregated_features[2,6]
   + A[8,2] * Conv_1_aggregated_features[2,6]
   + A[9,2] * Conv_1_aggregated_features[2,6] ] = 0
 Conv_1_averages_constraint[2,7]: [
   A[0,2] * Conv_1_aggregated_features[2,7]
   + A[1,2] * Conv_1_aggregated_features[2,7]
   - A[2,0] * Conv_0_Relu_ts[0,7] - A[2,1] * Conv_0_Relu_ts[1,7]
   - A[2,2] * Conv_0_Relu_ts[2,7]
   + A[2,2] * Conv_1_aggregated_features[2,7]
   - A[2,3] * Conv_0_Relu_ts[3,7] - A[2,4] * Conv_0_Relu_ts[4,7]
   - A[2,5] * Conv_0_Relu_ts[5,7] - A[2,6] * Conv_0_Relu_ts[6,7]
   - A[2,7] * Conv_0_Relu_ts[7,7] - A[2,8] * Conv_0_Relu_ts[8,7]
   - A[2,9] * Conv_0_Relu_ts[9,7]
   + A[3,2] * Conv_1_aggregated_features[2,7]
   + A[4,2] * Conv_1_aggregated_features[2,7]
   + A[5,2] * Conv_1_aggregated_features[2,7]
   + A[6,2] * Conv_1_aggregated_features[2,7]
   + A[7,2] * Conv_1_aggregated_features[2,7]
   + A[8,2] * Conv_1_aggregated_features[2,7]
   + A[9,2] * Conv_1_aggregated_features[2,7] ] = 0
 Conv_1_averages_constraint[2,8]: [
   A[0,2] * Conv_1_aggregated_features[2,8]
   + A[1,2] * Conv_1_aggregated_features[2,8]
   - A[2,0] * Conv_0_Relu_ts[0,8] - A[2,1] * Conv_0_Relu_ts[1,8]
   - A[2,2] * Conv_0_Relu_ts[2,8]
   + A[2,2] * Conv_1_aggregated_features[2,8]
   - A[2,3] * Conv_0_Relu_ts[3,8] - A[2,4] * Conv_0_Relu_ts[4,8]
   - A[2,5] * Conv_0_Relu_ts[5,8] - A[2,6] * Conv_0_Relu_ts[6,8]
   - A[2,7] * Conv_0_Relu_ts[7,8] - A[2,8] * Conv_0_Relu_ts[8,8]
   - A[2,9] * Conv_0_Relu_ts[9,8]
   + A[3,2] * Conv_1_aggregated_features[2,8]
   + A[4,2] * Conv_1_aggregated_features[2,8]
   + A[5,2] * Conv_1_aggregated_features[2,8]
   + A[6,2] * Conv_1_aggregated_features[2,8]
   + A[7,2] * Conv_1_aggregated_features[2,8]
   + A[8,2] * Conv_1_aggregated_features[2,8]
   + A[9,2] * Conv_1_aggregated_features[2,8] ] = 0
 Conv_1_averages_constraint[2,9]: [
   A[0,2] * Conv_1_aggregated_features[2,9]
   + A[1,2] * Conv_1_aggregated_features[2,9]
   - A[2,0] * Conv_0_Relu_ts[0,9] - A[2,1] * Conv_0_Relu_ts[1,9]
   - A[2,2] * Conv_0_Relu_ts[2,9]
   + A[2,2] * Conv_1_aggregated_features[2,9]
   - A[2,3] * Conv_0_Relu_ts[3,9] - A[2,4] * Conv_0_Relu_ts[4,9]
   - A[2,5] * Conv_0_Relu_ts[5,9] - A[2,6] * Conv_0_Relu_ts[6,9]
   - A[2,7] * Conv_0_Relu_ts[7,9] - A[2,8] * Conv_0_Relu_ts[8,9]
   - A[2,9] * Conv_0_Relu_ts[9,9]
   + A[3,2] * Conv_1_aggregated_features[2,9]
   + A[4,2] * Conv_1_aggregated_features[2,9]
   + A[5,2] * Conv_1_aggregated_features[2,9]
   + A[6,2] * Conv_1_aggregated_features[2,9]
   + A[7,2] * Conv_1_aggregated_features[2,9]
   + A[8,2] * Conv_1_aggregated_features[2,9]
   + A[9,2] * Conv_1_aggregated_features[2,9] ] = 0
 Conv_1_averages_constraint[2,10]: [
   A[0,2] * Conv_1_aggregated_features[2,10]
   + A[1,2] * Conv_1_aggregated_features[2,10]
   - A[2,0] * Conv_0_Relu_ts[0,10] - A[2,1] * Conv_0_Relu_ts[1,10]
   - A[2,2] * Conv_0_Relu_ts[2,10]
   + A[2,2] * Conv_1_aggregated_features[2,10]
   - A[2,3] * Conv_0_Relu_ts[3,10] - A[2,4] * Conv_0_Relu_ts[4,10]
   - A[2,5] * Conv_0_Relu_ts[5,10] - A[2,6] * Conv_0_Relu_ts[6,10]
   - A[2,7] * Conv_0_Relu_ts[7,10] - A[2,8] * Conv_0_Relu_ts[8,10]
   - A[2,9] * Conv_0_Relu_ts[9,10]
   + A[3,2] * Conv_1_aggregated_features[2,10]
   + A[4,2] * Conv_1_aggregated_features[2,10]
   + A[5,2] * Conv_1_aggregated_features[2,10]
   + A[6,2] * Conv_1_aggregated_features[2,10]
   + A[7,2] * Conv_1_aggregated_features[2,10]
   + A[8,2] * Conv_1_aggregated_features[2,10]
   + A[9,2] * Conv_1_aggregated_features[2,10] ] = 0
 Conv_1_averages_constraint[2,11]: [
   A[0,2] * Conv_1_aggregated_features[2,11]
   + A[1,2] * Conv_1_aggregated_features[2,11]
   - A[2,0] * Conv_0_Relu_ts[0,11] - A[2,1] * Conv_0_Relu_ts[1,11]
   - A[2,2] * Conv_0_Relu_ts[2,11]
   + A[2,2] * Conv_1_aggregated_features[2,11]
   - A[2,3] * Conv_0_Relu_ts[3,11] - A[2,4] * Conv_0_Relu_ts[4,11]
   - A[2,5] * Conv_0_Relu_ts[5,11] - A[2,6] * Conv_0_Relu_ts[6,11]
   - A[2,7] * Conv_0_Relu_ts[7,11] - A[2,8] * Conv_0_Relu_ts[8,11]
   - A[2,9] * Conv_0_Relu_ts[9,11]
   + A[3,2] * Conv_1_aggregated_features[2,11]
   + A[4,2] * Conv_1_aggregated_features[2,11]
   + A[5,2] * Conv_1_aggregated_features[2,11]
   + A[6,2] * Conv_1_aggregated_features[2,11]
   + A[7,2] * Conv_1_aggregated_features[2,11]
   + A[8,2] * Conv_1_aggregated_features[2,11]
   + A[9,2] * Conv_1_aggregated_features[2,11] ] = 0
 Conv_1_averages_constraint[2,12]: [
   A[0,2] * Conv_1_aggregated_features[2,12]
   + A[1,2] * Conv_1_aggregated_features[2,12]
   - A[2,0] * Conv_0_Relu_ts[0,12] - A[2,1] * Conv_0_Relu_ts[1,12]
   - A[2,2] * Conv_0_Relu_ts[2,12]
   + A[2,2] * Conv_1_aggregated_features[2,12]
   - A[2,3] * Conv_0_Relu_ts[3,12] - A[2,4] * Conv_0_Relu_ts[4,12]
   - A[2,5] * Conv_0_Relu_ts[5,12] - A[2,6] * Conv_0_Relu_ts[6,12]
   - A[2,7] * Conv_0_Relu_ts[7,12] - A[2,8] * Conv_0_Relu_ts[8,12]
   - A[2,9] * Conv_0_Relu_ts[9,12]
   + A[3,2] * Conv_1_aggregated_features[2,12]
   + A[4,2] * Conv_1_aggregated_features[2,12]
   + A[5,2] * Conv_1_aggregated_features[2,12]
   + A[6,2] * Conv_1_aggregated_features[2,12]
   + A[7,2] * Conv_1_aggregated_features[2,12]
   + A[8,2] * Conv_1_aggregated_features[2,12]
   + A[9,2] * Conv_1_aggregated_features[2,12] ] = 0
 Conv_1_averages_constraint[2,13]: [
   A[0,2] * Conv_1_aggregated_features[2,13]
   + A[1,2] * Conv_1_aggregated_features[2,13]
   - A[2,0] * Conv_0_Relu_ts[0,13] - A[2,1] * Conv_0_Relu_ts[1,13]
   - A[2,2] * Conv_0_Relu_ts[2,13]
   + A[2,2] * Conv_1_aggregated_features[2,13]
   - A[2,3] * Conv_0_Relu_ts[3,13] - A[2,4] * Conv_0_Relu_ts[4,13]
   - A[2,5] * Conv_0_Relu_ts[5,13] - A[2,6] * Conv_0_Relu_ts[6,13]
   - A[2,7] * Conv_0_Relu_ts[7,13] - A[2,8] * Conv_0_Relu_ts[8,13]
   - A[2,9] * Conv_0_Relu_ts[9,13]
   + A[3,2] * Conv_1_aggregated_features[2,13]
   + A[4,2] * Conv_1_aggregated_features[2,13]
   + A[5,2] * Conv_1_aggregated_features[2,13]
   + A[6,2] * Conv_1_aggregated_features[2,13]
   + A[7,2] * Conv_1_aggregated_features[2,13]
   + A[8,2] * Conv_1_aggregated_features[2,13]
   + A[9,2] * Conv_1_aggregated_features[2,13] ] = 0
 Conv_1_averages_constraint[2,14]: [
   A[0,2] * Conv_1_aggregated_features[2,14]
   + A[1,2] * Conv_1_aggregated_features[2,14]
   - A[2,0] * Conv_0_Relu_ts[0,14] - A[2,1] * Conv_0_Relu_ts[1,14]
   - A[2,2] * Conv_0_Relu_ts[2,14]
   + A[2,2] * Conv_1_aggregated_features[2,14]
   - A[2,3] * Conv_0_Relu_ts[3,14] - A[2,4] * Conv_0_Relu_ts[4,14]
   - A[2,5] * Conv_0_Relu_ts[5,14] - A[2,6] * Conv_0_Relu_ts[6,14]
   - A[2,7] * Conv_0_Relu_ts[7,14] - A[2,8] * Conv_0_Relu_ts[8,14]
   - A[2,9] * Conv_0_Relu_ts[9,14]
   + A[3,2] * Conv_1_aggregated_features[2,14]
   + A[4,2] * Conv_1_aggregated_features[2,14]
   + A[5,2] * Conv_1_aggregated_features[2,14]
   + A[6,2] * Conv_1_aggregated_features[2,14]
   + A[7,2] * Conv_1_aggregated_features[2,14]
   + A[8,2] * Conv_1_aggregated_features[2,14]
   + A[9,2] * Conv_1_aggregated_features[2,14] ] = 0
 Conv_1_averages_constraint[2,15]: [
   A[0,2] * Conv_1_aggregated_features[2,15]
   + A[1,2] * Conv_1_aggregated_features[2,15]
   - A[2,0] * Conv_0_Relu_ts[0,15] - A[2,1] * Conv_0_Relu_ts[1,15]
   - A[2,2] * Conv_0_Relu_ts[2,15]
   + A[2,2] * Conv_1_aggregated_features[2,15]
   - A[2,3] * Conv_0_Relu_ts[3,15] - A[2,4] * Conv_0_Relu_ts[4,15]
   - A[2,5] * Conv_0_Relu_ts[5,15] - A[2,6] * Conv_0_Relu_ts[6,15]
   - A[2,7] * Conv_0_Relu_ts[7,15] - A[2,8] * Conv_0_Relu_ts[8,15]
   - A[2,9] * Conv_0_Relu_ts[9,15]
   + A[3,2] * Conv_1_aggregated_features[2,15]
   + A[4,2] * Conv_1_aggregated_features[2,15]
   + A[5,2] * Conv_1_aggregated_features[2,15]
   + A[6,2] * Conv_1_aggregated_features[2,15]
   + A[7,2] * Conv_1_aggregated_features[2,15]
   + A[8,2] * Conv_1_aggregated_features[2,15]
   + A[9,2] * Conv_1_aggregated_features[2,15] ] = 0
 Conv_1_averages_constraint[3,0]: [
   A[0,3] * Conv_1_aggregated_features[3,0]
   + A[1,3] * Conv_1_aggregated_features[3,0]
   + A[2,3] * Conv_1_aggregated_features[3,0]
   - A[3,0] * Conv_0_Relu_ts[0,0] - A[3,1] * Conv_0_Relu_ts[1,0]
   - A[3,2] * Conv_0_Relu_ts[2,0] - A[3,3] * Conv_0_Relu_ts[3,0]
   + A[3,3] * Conv_1_aggregated_features[3,0]
   - A[3,4] * Conv_0_Relu_ts[4,0] - A[3,5] * Conv_0_Relu_ts[5,0]
   - A[3,6] * Conv_0_Relu_ts[6,0] - A[3,7] * Conv_0_Relu_ts[7,0]
   - A[3,8] * Conv_0_Relu_ts[8,0] - A[3,9] * Conv_0_Relu_ts[9,0]
   + A[4,3] * Conv_1_aggregated_features[3,0]
   + A[5,3] * Conv_1_aggregated_features[3,0]
   + A[6,3] * Conv_1_aggregated_features[3,0]
   + A[7,3] * Conv_1_aggregated_features[3,0]
   + A[8,3] * Conv_1_aggregated_features[3,0]
   + A[9,3] * Conv_1_aggregated_features[3,0] ] = 0
 Conv_1_averages_constraint[3,1]: [
   A[0,3] * Conv_1_aggregated_features[3,1]
   + A[1,3] * Conv_1_aggregated_features[3,1]
   + A[2,3] * Conv_1_aggregated_features[3,1]
   - A[3,0] * Conv_0_Relu_ts[0,1] - A[3,1] * Conv_0_Relu_ts[1,1]
   - A[3,2] * Conv_0_Relu_ts[2,1] - A[3,3] * Conv_0_Relu_ts[3,1]
   + A[3,3] * Conv_1_aggregated_features[3,1]
   - A[3,4] * Conv_0_Relu_ts[4,1] - A[3,5] * Conv_0_Relu_ts[5,1]
   - A[3,6] * Conv_0_Relu_ts[6,1] - A[3,7] * Conv_0_Relu_ts[7,1]
   - A[3,8] * Conv_0_Relu_ts[8,1] - A[3,9] * Conv_0_Relu_ts[9,1]
   + A[4,3] * Conv_1_aggregated_features[3,1]
   + A[5,3] * Conv_1_aggregated_features[3,1]
   + A[6,3] * Conv_1_aggregated_features[3,1]
   + A[7,3] * Conv_1_aggregated_features[3,1]
   + A[8,3] * Conv_1_aggregated_features[3,1]
   + A[9,3] * Conv_1_aggregated_features[3,1] ] = 0
 Conv_1_averages_constraint[3,2]: [
   A[0,3] * Conv_1_aggregated_features[3,2]
   + A[1,3] * Conv_1_aggregated_features[3,2]
   + A[2,3] * Conv_1_aggregated_features[3,2]
   - A[3,0] * Conv_0_Relu_ts[0,2] - A[3,1] * Conv_0_Relu_ts[1,2]
   - A[3,2] * Conv_0_Relu_ts[2,2] - A[3,3] * Conv_0_Relu_ts[3,2]
   + A[3,3] * Conv_1_aggregated_features[3,2]
   - A[3,4] * Conv_0_Relu_ts[4,2] - A[3,5] * Conv_0_Relu_ts[5,2]
   - A[3,6] * Conv_0_Relu_ts[6,2] - A[3,7] * Conv_0_Relu_ts[7,2]
   - A[3,8] * Conv_0_Relu_ts[8,2] - A[3,9] * Conv_0_Relu_ts[9,2]
   + A[4,3] * Conv_1_aggregated_features[3,2]
   + A[5,3] * Conv_1_aggregated_features[3,2]
   + A[6,3] * Conv_1_aggregated_features[3,2]
   + A[7,3] * Conv_1_aggregated_features[3,2]
   + A[8,3] * Conv_1_aggregated_features[3,2]
   + A[9,3] * Conv_1_aggregated_features[3,2] ] = 0
 Conv_1_averages_constraint[3,3]: [
   A[0,3] * Conv_1_aggregated_features[3,3]
   + A[1,3] * Conv_1_aggregated_features[3,3]
   + A[2,3] * Conv_1_aggregated_features[3,3]
   - A[3,0] * Conv_0_Relu_ts[0,3] - A[3,1] * Conv_0_Relu_ts[1,3]
   - A[3,2] * Conv_0_Relu_ts[2,3] - A[3,3] * Conv_0_Relu_ts[3,3]
   + A[3,3] * Conv_1_aggregated_features[3,3]
   - A[3,4] * Conv_0_Relu_ts[4,3] - A[3,5] * Conv_0_Relu_ts[5,3]
   - A[3,6] * Conv_0_Relu_ts[6,3] - A[3,7] * Conv_0_Relu_ts[7,3]
   - A[3,8] * Conv_0_Relu_ts[8,3] - A[3,9] * Conv_0_Relu_ts[9,3]
   + A[4,3] * Conv_1_aggregated_features[3,3]
   + A[5,3] * Conv_1_aggregated_features[3,3]
   + A[6,3] * Conv_1_aggregated_features[3,3]
   + A[7,3] * Conv_1_aggregated_features[3,3]
   + A[8,3] * Conv_1_aggregated_features[3,3]
   + A[9,3] * Conv_1_aggregated_features[3,3] ] = 0
 Conv_1_averages_constraint[3,4]: [
   A[0,3] * Conv_1_aggregated_features[3,4]
   + A[1,3] * Conv_1_aggregated_features[3,4]
   + A[2,3] * Conv_1_aggregated_features[3,4]
   - A[3,0] * Conv_0_Relu_ts[0,4] - A[3,1] * Conv_0_Relu_ts[1,4]
   - A[3,2] * Conv_0_Relu_ts[2,4] - A[3,3] * Conv_0_Relu_ts[3,4]
   + A[3,3] * Conv_1_aggregated_features[3,4]
   - A[3,4] * Conv_0_Relu_ts[4,4] - A[3,5] * Conv_0_Relu_ts[5,4]
   - A[3,6] * Conv_0_Relu_ts[6,4] - A[3,7] * Conv_0_Relu_ts[7,4]
   - A[3,8] * Conv_0_Relu_ts[8,4] - A[3,9] * Conv_0_Relu_ts[9,4]
   + A[4,3] * Conv_1_aggregated_features[3,4]
   + A[5,3] * Conv_1_aggregated_features[3,4]
   + A[6,3] * Conv_1_aggregated_features[3,4]
   + A[7,3] * Conv_1_aggregated_features[3,4]
   + A[8,3] * Conv_1_aggregated_features[3,4]
   + A[9,3] * Conv_1_aggregated_features[3,4] ] = 0
 Conv_1_averages_constraint[3,5]: [
   A[0,3] * Conv_1_aggregated_features[3,5]
   + A[1,3] * Conv_1_aggregated_features[3,5]
   + A[2,3] * Conv_1_aggregated_features[3,5]
   - A[3,0] * Conv_0_Relu_ts[0,5] - A[3,1] * Conv_0_Relu_ts[1,5]
   - A[3,2] * Conv_0_Relu_ts[2,5] - A[3,3] * Conv_0_Relu_ts[3,5]
   + A[3,3] * Conv_1_aggregated_features[3,5]
   - A[3,4] * Conv_0_Relu_ts[4,5] - A[3,5] * Conv_0_Relu_ts[5,5]
   - A[3,6] * Conv_0_Relu_ts[6,5] - A[3,7] * Conv_0_Relu_ts[7,5]
   - A[3,8] * Conv_0_Relu_ts[8,5] - A[3,9] * Conv_0_Relu_ts[9,5]
   + A[4,3] * Conv_1_aggregated_features[3,5]
   + A[5,3] * Conv_1_aggregated_features[3,5]
   + A[6,3] * Conv_1_aggregated_features[3,5]
   + A[7,3] * Conv_1_aggregated_features[3,5]
   + A[8,3] * Conv_1_aggregated_features[3,5]
   + A[9,3] * Conv_1_aggregated_features[3,5] ] = 0
 Conv_1_averages_constraint[3,6]: [
   A[0,3] * Conv_1_aggregated_features[3,6]
   + A[1,3] * Conv_1_aggregated_features[3,6]
   + A[2,3] * Conv_1_aggregated_features[3,6]
   - A[3,0] * Conv_0_Relu_ts[0,6] - A[3,1] * Conv_0_Relu_ts[1,6]
   - A[3,2] * Conv_0_Relu_ts[2,6] - A[3,3] * Conv_0_Relu_ts[3,6]
   + A[3,3] * Conv_1_aggregated_features[3,6]
   - A[3,4] * Conv_0_Relu_ts[4,6] - A[3,5] * Conv_0_Relu_ts[5,6]
   - A[3,6] * Conv_0_Relu_ts[6,6] - A[3,7] * Conv_0_Relu_ts[7,6]
   - A[3,8] * Conv_0_Relu_ts[8,6] - A[3,9] * Conv_0_Relu_ts[9,6]
   + A[4,3] * Conv_1_aggregated_features[3,6]
   + A[5,3] * Conv_1_aggregated_features[3,6]
   + A[6,3] * Conv_1_aggregated_features[3,6]
   + A[7,3] * Conv_1_aggregated_features[3,6]
   + A[8,3] * Conv_1_aggregated_features[3,6]
   + A[9,3] * Conv_1_aggregated_features[3,6] ] = 0
 Conv_1_averages_constraint[3,7]: [
   A[0,3] * Conv_1_aggregated_features[3,7]
   + A[1,3] * Conv_1_aggregated_features[3,7]
   + A[2,3] * Conv_1_aggregated_features[3,7]
   - A[3,0] * Conv_0_Relu_ts[0,7] - A[3,1] * Conv_0_Relu_ts[1,7]
   - A[3,2] * Conv_0_Relu_ts[2,7] - A[3,3] * Conv_0_Relu_ts[3,7]
   + A[3,3] * Conv_1_aggregated_features[3,7]
   - A[3,4] * Conv_0_Relu_ts[4,7] - A[3,5] * Conv_0_Relu_ts[5,7]
   - A[3,6] * Conv_0_Relu_ts[6,7] - A[3,7] * Conv_0_Relu_ts[7,7]
   - A[3,8] * Conv_0_Relu_ts[8,7] - A[3,9] * Conv_0_Relu_ts[9,7]
   + A[4,3] * Conv_1_aggregated_features[3,7]
   + A[5,3] * Conv_1_aggregated_features[3,7]
   + A[6,3] * Conv_1_aggregated_features[3,7]
   + A[7,3] * Conv_1_aggregated_features[3,7]
   + A[8,3] * Conv_1_aggregated_features[3,7]
   + A[9,3] * Conv_1_aggregated_features[3,7] ] = 0
 Conv_1_averages_constraint[3,8]: [
   A[0,3] * Conv_1_aggregated_features[3,8]
   + A[1,3] * Conv_1_aggregated_features[3,8]
   + A[2,3] * Conv_1_aggregated_features[3,8]
   - A[3,0] * Conv_0_Relu_ts[0,8] - A[3,1] * Conv_0_Relu_ts[1,8]
   - A[3,2] * Conv_0_Relu_ts[2,8] - A[3,3] * Conv_0_Relu_ts[3,8]
   + A[3,3] * Conv_1_aggregated_features[3,8]
   - A[3,4] * Conv_0_Relu_ts[4,8] - A[3,5] * Conv_0_Relu_ts[5,8]
   - A[3,6] * Conv_0_Relu_ts[6,8] - A[3,7] * Conv_0_Relu_ts[7,8]
   - A[3,8] * Conv_0_Relu_ts[8,8] - A[3,9] * Conv_0_Relu_ts[9,8]
   + A[4,3] * Conv_1_aggregated_features[3,8]
   + A[5,3] * Conv_1_aggregated_features[3,8]
   + A[6,3] * Conv_1_aggregated_features[3,8]
   + A[7,3] * Conv_1_aggregated_features[3,8]
   + A[8,3] * Conv_1_aggregated_features[3,8]
   + A[9,3] * Conv_1_aggregated_features[3,8] ] = 0
 Conv_1_averages_constraint[3,9]: [
   A[0,3] * Conv_1_aggregated_features[3,9]
   + A[1,3] * Conv_1_aggregated_features[3,9]
   + A[2,3] * Conv_1_aggregated_features[3,9]
   - A[3,0] * Conv_0_Relu_ts[0,9] - A[3,1] * Conv_0_Relu_ts[1,9]
   - A[3,2] * Conv_0_Relu_ts[2,9] - A[3,3] * Conv_0_Relu_ts[3,9]
   + A[3,3] * Conv_1_aggregated_features[3,9]
   - A[3,4] * Conv_0_Relu_ts[4,9] - A[3,5] * Conv_0_Relu_ts[5,9]
   - A[3,6] * Conv_0_Relu_ts[6,9] - A[3,7] * Conv_0_Relu_ts[7,9]
   - A[3,8] * Conv_0_Relu_ts[8,9] - A[3,9] * Conv_0_Relu_ts[9,9]
   + A[4,3] * Conv_1_aggregated_features[3,9]
   + A[5,3] * Conv_1_aggregated_features[3,9]
   + A[6,3] * Conv_1_aggregated_features[3,9]
   + A[7,3] * Conv_1_aggregated_features[3,9]
   + A[8,3] * Conv_1_aggregated_features[3,9]
   + A[9,3] * Conv_1_aggregated_features[3,9] ] = 0
 Conv_1_averages_constraint[3,10]: [
   A[0,3] * Conv_1_aggregated_features[3,10]
   + A[1,3] * Conv_1_aggregated_features[3,10]
   + A[2,3] * Conv_1_aggregated_features[3,10]
   - A[3,0] * Conv_0_Relu_ts[0,10] - A[3,1] * Conv_0_Relu_ts[1,10]
   - A[3,2] * Conv_0_Relu_ts[2,10] - A[3,3] * Conv_0_Relu_ts[3,10]
   + A[3,3] * Conv_1_aggregated_features[3,10]
   - A[3,4] * Conv_0_Relu_ts[4,10] - A[3,5] * Conv_0_Relu_ts[5,10]
   - A[3,6] * Conv_0_Relu_ts[6,10] - A[3,7] * Conv_0_Relu_ts[7,10]
   - A[3,8] * Conv_0_Relu_ts[8,10] - A[3,9] * Conv_0_Relu_ts[9,10]
   + A[4,3] * Conv_1_aggregated_features[3,10]
   + A[5,3] * Conv_1_aggregated_features[3,10]
   + A[6,3] * Conv_1_aggregated_features[3,10]
   + A[7,3] * Conv_1_aggregated_features[3,10]
   + A[8,3] * Conv_1_aggregated_features[3,10]
   + A[9,3] * Conv_1_aggregated_features[3,10] ] = 0
 Conv_1_averages_constraint[3,11]: [
   A[0,3] * Conv_1_aggregated_features[3,11]
   + A[1,3] * Conv_1_aggregated_features[3,11]
   + A[2,3] * Conv_1_aggregated_features[3,11]
   - A[3,0] * Conv_0_Relu_ts[0,11] - A[3,1] * Conv_0_Relu_ts[1,11]
   - A[3,2] * Conv_0_Relu_ts[2,11] - A[3,3] * Conv_0_Relu_ts[3,11]
   + A[3,3] * Conv_1_aggregated_features[3,11]
   - A[3,4] * Conv_0_Relu_ts[4,11] - A[3,5] * Conv_0_Relu_ts[5,11]
   - A[3,6] * Conv_0_Relu_ts[6,11] - A[3,7] * Conv_0_Relu_ts[7,11]
   - A[3,8] * Conv_0_Relu_ts[8,11] - A[3,9] * Conv_0_Relu_ts[9,11]
   + A[4,3] * Conv_1_aggregated_features[3,11]
   + A[5,3] * Conv_1_aggregated_features[3,11]
   + A[6,3] * Conv_1_aggregated_features[3,11]
   + A[7,3] * Conv_1_aggregated_features[3,11]
   + A[8,3] * Conv_1_aggregated_features[3,11]
   + A[9,3] * Conv_1_aggregated_features[3,11] ] = 0
 Conv_1_averages_constraint[3,12]: [
   A[0,3] * Conv_1_aggregated_features[3,12]
   + A[1,3] * Conv_1_aggregated_features[3,12]
   + A[2,3] * Conv_1_aggregated_features[3,12]
   - A[3,0] * Conv_0_Relu_ts[0,12] - A[3,1] * Conv_0_Relu_ts[1,12]
   - A[3,2] * Conv_0_Relu_ts[2,12] - A[3,3] * Conv_0_Relu_ts[3,12]
   + A[3,3] * Conv_1_aggregated_features[3,12]
   - A[3,4] * Conv_0_Relu_ts[4,12] - A[3,5] * Conv_0_Relu_ts[5,12]
   - A[3,6] * Conv_0_Relu_ts[6,12] - A[3,7] * Conv_0_Relu_ts[7,12]
   - A[3,8] * Conv_0_Relu_ts[8,12] - A[3,9] * Conv_0_Relu_ts[9,12]
   + A[4,3] * Conv_1_aggregated_features[3,12]
   + A[5,3] * Conv_1_aggregated_features[3,12]
   + A[6,3] * Conv_1_aggregated_features[3,12]
   + A[7,3] * Conv_1_aggregated_features[3,12]
   + A[8,3] * Conv_1_aggregated_features[3,12]
   + A[9,3] * Conv_1_aggregated_features[3,12] ] = 0
 Conv_1_averages_constraint[3,13]: [
   A[0,3] * Conv_1_aggregated_features[3,13]
   + A[1,3] * Conv_1_aggregated_features[3,13]
   + A[2,3] * Conv_1_aggregated_features[3,13]
   - A[3,0] * Conv_0_Relu_ts[0,13] - A[3,1] * Conv_0_Relu_ts[1,13]
   - A[3,2] * Conv_0_Relu_ts[2,13] - A[3,3] * Conv_0_Relu_ts[3,13]
   + A[3,3] * Conv_1_aggregated_features[3,13]
   - A[3,4] * Conv_0_Relu_ts[4,13] - A[3,5] * Conv_0_Relu_ts[5,13]
   - A[3,6] * Conv_0_Relu_ts[6,13] - A[3,7] * Conv_0_Relu_ts[7,13]
   - A[3,8] * Conv_0_Relu_ts[8,13] - A[3,9] * Conv_0_Relu_ts[9,13]
   + A[4,3] * Conv_1_aggregated_features[3,13]
   + A[5,3] * Conv_1_aggregated_features[3,13]
   + A[6,3] * Conv_1_aggregated_features[3,13]
   + A[7,3] * Conv_1_aggregated_features[3,13]
   + A[8,3] * Conv_1_aggregated_features[3,13]
   + A[9,3] * Conv_1_aggregated_features[3,13] ] = 0
 Conv_1_averages_constraint[3,14]: [
   A[0,3] * Conv_1_aggregated_features[3,14]
   + A[1,3] * Conv_1_aggregated_features[3,14]
   + A[2,3] * Conv_1_aggregated_features[3,14]
   - A[3,0] * Conv_0_Relu_ts[0,14] - A[3,1] * Conv_0_Relu_ts[1,14]
   - A[3,2] * Conv_0_Relu_ts[2,14] - A[3,3] * Conv_0_Relu_ts[3,14]
   + A[3,3] * Conv_1_aggregated_features[3,14]
   - A[3,4] * Conv_0_Relu_ts[4,14] - A[3,5] * Conv_0_Relu_ts[5,14]
   - A[3,6] * Conv_0_Relu_ts[6,14] - A[3,7] * Conv_0_Relu_ts[7,14]
   - A[3,8] * Conv_0_Relu_ts[8,14] - A[3,9] * Conv_0_Relu_ts[9,14]
   + A[4,3] * Conv_1_aggregated_features[3,14]
   + A[5,3] * Conv_1_aggregated_features[3,14]
   + A[6,3] * Conv_1_aggregated_features[3,14]
   + A[7,3] * Conv_1_aggregated_features[3,14]
   + A[8,3] * Conv_1_aggregated_features[3,14]
   + A[9,3] * Conv_1_aggregated_features[3,14] ] = 0
 Conv_1_averages_constraint[3,15]: [
   A[0,3] * Conv_1_aggregated_features[3,15]
   + A[1,3] * Conv_1_aggregated_features[3,15]
   + A[2,3] * Conv_1_aggregated_features[3,15]
   - A[3,0] * Conv_0_Relu_ts[0,15] - A[3,1] * Conv_0_Relu_ts[1,15]
   - A[3,2] * Conv_0_Relu_ts[2,15] - A[3,3] * Conv_0_Relu_ts[3,15]
   + A[3,3] * Conv_1_aggregated_features[3,15]
   - A[3,4] * Conv_0_Relu_ts[4,15] - A[3,5] * Conv_0_Relu_ts[5,15]
   - A[3,6] * Conv_0_Relu_ts[6,15] - A[3,7] * Conv_0_Relu_ts[7,15]
   - A[3,8] * Conv_0_Relu_ts[8,15] - A[3,9] * Conv_0_Relu_ts[9,15]
   + A[4,3] * Conv_1_aggregated_features[3,15]
   + A[5,3] * Conv_1_aggregated_features[3,15]
   + A[6,3] * Conv_1_aggregated_features[3,15]
   + A[7,3] * Conv_1_aggregated_features[3,15]
   + A[8,3] * Conv_1_aggregated_features[3,15]
   + A[9,3] * Conv_1_aggregated_features[3,15] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,0] - A[4,8] * Conv_0_Relu_ts[8,0]
   - A[4,9] * Conv_0_Relu_ts[9,0]
   + A[5,4] * Conv_1_aggregated_features[4,0]
   + A[6,4] * Conv_1_aggregated_features[4,0]
   + A[7,4] * Conv_1_aggregated_features[4,0]
   + A[8,4] * Conv_1_aggregated_features[4,0]
   + A[9,4] * Conv_1_aggregated_features[4,0] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,1] - A[4,8] * Conv_0_Relu_ts[8,1]
   - A[4,9] * Conv_0_Relu_ts[9,1]
   + A[5,4] * Conv_1_aggregated_features[4,1]
   + A[6,4] * Conv_1_aggregated_features[4,1]
   + A[7,4] * Conv_1_aggregated_features[4,1]
   + A[8,4] * Conv_1_aggregated_features[4,1]
   + A[9,4] * Conv_1_aggregated_features[4,1] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,2] - A[4,8] * Conv_0_Relu_ts[8,2]
   - A[4,9] * Conv_0_Relu_ts[9,2]
   + A[5,4] * Conv_1_aggregated_features[4,2]
   + A[6,4] * Conv_1_aggregated_features[4,2]
   + A[7,4] * Conv_1_aggregated_features[4,2]
   + A[8,4] * Conv_1_aggregated_features[4,2]
   + A[9,4] * Conv_1_aggregated_features[4,2] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,3] - A[4,8] * Conv_0_Relu_ts[8,3]
   - A[4,9] * Conv_0_Relu_ts[9,3]
   + A[5,4] * Conv_1_aggregated_features[4,3]
   + A[6,4] * Conv_1_aggregated_features[4,3]
   + A[7,4] * Conv_1_aggregated_features[4,3]
   + A[8,4] * Conv_1_aggregated_features[4,3]
   + A[9,4] * Conv_1_aggregated_features[4,3] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,4] - A[4,8] * Conv_0_Relu_ts[8,4]
   - A[4,9] * Conv_0_Relu_ts[9,4]
   + A[5,4] * Conv_1_aggregated_features[4,4]
   + A[6,4] * Conv_1_aggregated_features[4,4]
   + A[7,4] * Conv_1_aggregated_features[4,4]
   + A[8,4] * Conv_1_aggregated_features[4,4]
   + A[9,4] * Conv_1_aggregated_features[4,4] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,5] - A[4,8] * Conv_0_Relu_ts[8,5]
   - A[4,9] * Conv_0_Relu_ts[9,5]
   + A[5,4] * Conv_1_aggregated_features[4,5]
   + A[6,4] * Conv_1_aggregated_features[4,5]
   + A[7,4] * Conv_1_aggregated_features[4,5]
   + A[8,4] * Conv_1_aggregated_features[4,5]
   + A[9,4] * Conv_1_aggregated_features[4,5] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,6] - A[4,8] * Conv_0_Relu_ts[8,6]
   - A[4,9] * Conv_0_Relu_ts[9,6]
   + A[5,4] * Conv_1_aggregated_features[4,6]
   + A[6,4] * Conv_1_aggregated_features[4,6]
   + A[7,4] * Conv_1_aggregated_features[4,6]
   + A[8,4] * Conv_1_aggregated_features[4,6]
   + A[9,4] * Conv_1_aggregated_features[4,6] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,7] - A[4,8] * Conv_0_Relu_ts[8,7]
   - A[4,9] * Conv_0_Relu_ts[9,7]
   + A[5,4] * Conv_1_aggregated_features[4,7]
   + A[6,4] * Conv_1_aggregated_features[4,7]
   + A[7,4] * Conv_1_aggregated_features[4,7]
   + A[8,4] * Conv_1_aggregated_features[4,7]
   + A[9,4] * Conv_1_aggregated_features[4,7] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,8] - A[4,8] * Conv_0_Relu_ts[8,8]
   - A[4,9] * Conv_0_Relu_ts[9,8]
   + A[5,4] * Conv_1_aggregated_features[4,8]
   + A[6,4] * Conv_1_aggregated_features[4,8]
   + A[7,4] * Conv_1_aggregated_features[4,8]
   + A[8,4] * Conv_1_aggregated_features[4,8]
   + A[9,4] * Conv_1_aggregated_features[4,8] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,9] - A[4,8] * Conv_0_Relu_ts[8,9]
   - A[4,9] * Conv_0_Relu_ts[9,9]
   + A[5,4] * Conv_1_aggregated_features[4,9]
   + A[6,4] * Conv_1_aggregated_features[4,9]
   + A[7,4] * Conv_1_aggregated_features[4,9]
   + A[8,4] * Conv_1_aggregated_features[4,9]
   + A[9,4] * Conv_1_aggregated_features[4,9] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,10] - A[4,8] * Conv_0_Relu_ts[8,10]
   - A[4,9] * Conv_0_Relu_ts[9,10]
   + A[5,4] * Conv_1_aggregated_features[4,10]
   + A[6,4] * Conv_1_aggregated_features[4,10]
   + A[7,4] * Conv_1_aggregated_features[4,10]
   + A[8,4] * Conv_1_aggregated_features[4,10]
   + A[9,4] * Conv_1_aggregated_features[4,10] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,11] - A[4,8] * Conv_0_Relu_ts[8,11]
   - A[4,9] * Conv_0_Relu_ts[9,11]
   + A[5,4] * Conv_1_aggregated_features[4,11]
   + A[6,4] * Conv_1_aggregated_features[4,11]
   + A[7,4] * Conv_1_aggregated_features[4,11]
   + A[8,4] * Conv_1_aggregated_features[4,11]
   + A[9,4] * Conv_1_aggregated_features[4,11] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,12] - A[4,8] * Conv_0_Relu_ts[8,12]
   - A[4,9] * Conv_0_Relu_ts[9,12]
   + A[5,4] * Conv_1_aggregated_features[4,12]
   + A[6,4] * Conv_1_aggregated_features[4,12]
   + A[7,4] * Conv_1_aggregated_features[4,12]
   + A[8,4] * Conv_1_aggregated_features[4,12]
   + A[9,4] * Conv_1_aggregated_features[4,12] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,13] - A[4,8] * Conv_0_Relu_ts[8,13]
   - A[4,9] * Conv_0_Relu_ts[9,13]
   + A[5,4] * Conv_1_aggregated_features[4,13]
   + A[6,4] * Conv_1_aggregated_features[4,13]
   + A[7,4] * Conv_1_aggregated_features[4,13]
   + A[8,4] * Conv_1_aggregated_features[4,13]
   + A[9,4] * Conv_1_aggregated_features[4,13] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,14] - A[4,8] * Conv_0_Relu_ts[8,14]
   - A[4,9] * Conv_0_Relu_ts[9,14]
   + A[5,4] * Conv_1_aggregated_features[4,14]
   + A[6,4] * Conv_1_aggregated_features[4,14]
   + A[7,4] * Conv_1_aggregated_features[4,14]
   + A[8,4] * Conv_1_aggregated_features[4,14]
   + A[9,4] * Conv_1_aggregated_features[4,14] ] = 0
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
   - A[4,7] * Conv_0_Relu_ts[7,15] - A[4,8] * Conv_0_Relu_ts[8,15]
   - A[4,9] * Conv_0_Relu_ts[9,15]
   + A[5,4] * Conv_1_aggregated_features[4,15]
   + A[6,4] * Conv_1_aggregated_features[4,15]
   + A[7,4] * Conv_1_aggregated_features[4,15]
   + A[8,4] * Conv_1_aggregated_features[4,15]
   + A[9,4] * Conv_1_aggregated_features[4,15] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,0] - A[5,9] * Conv_0_Relu_ts[9,0]
   + A[6,5] * Conv_1_aggregated_features[5,0]
   + A[7,5] * Conv_1_aggregated_features[5,0]
   + A[8,5] * Conv_1_aggregated_features[5,0]
   + A[9,5] * Conv_1_aggregated_features[5,0] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,1] - A[5,9] * Conv_0_Relu_ts[9,1]
   + A[6,5] * Conv_1_aggregated_features[5,1]
   + A[7,5] * Conv_1_aggregated_features[5,1]
   + A[8,5] * Conv_1_aggregated_features[5,1]
   + A[9,5] * Conv_1_aggregated_features[5,1] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,2] - A[5,9] * Conv_0_Relu_ts[9,2]
   + A[6,5] * Conv_1_aggregated_features[5,2]
   + A[7,5] * Conv_1_aggregated_features[5,2]
   + A[8,5] * Conv_1_aggregated_features[5,2]
   + A[9,5] * Conv_1_aggregated_features[5,2] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,3] - A[5,9] * Conv_0_Relu_ts[9,3]
   + A[6,5] * Conv_1_aggregated_features[5,3]
   + A[7,5] * Conv_1_aggregated_features[5,3]
   + A[8,5] * Conv_1_aggregated_features[5,3]
   + A[9,5] * Conv_1_aggregated_features[5,3] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,4] - A[5,9] * Conv_0_Relu_ts[9,4]
   + A[6,5] * Conv_1_aggregated_features[5,4]
   + A[7,5] * Conv_1_aggregated_features[5,4]
   + A[8,5] * Conv_1_aggregated_features[5,4]
   + A[9,5] * Conv_1_aggregated_features[5,4] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,5] - A[5,9] * Conv_0_Relu_ts[9,5]
   + A[6,5] * Conv_1_aggregated_features[5,5]
   + A[7,5] * Conv_1_aggregated_features[5,5]
   + A[8,5] * Conv_1_aggregated_features[5,5]
   + A[9,5] * Conv_1_aggregated_features[5,5] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,6] - A[5,9] * Conv_0_Relu_ts[9,6]
   + A[6,5] * Conv_1_aggregated_features[5,6]
   + A[7,5] * Conv_1_aggregated_features[5,6]
   + A[8,5] * Conv_1_aggregated_features[5,6]
   + A[9,5] * Conv_1_aggregated_features[5,6] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,7] - A[5,9] * Conv_0_Relu_ts[9,7]
   + A[6,5] * Conv_1_aggregated_features[5,7]
   + A[7,5] * Conv_1_aggregated_features[5,7]
   + A[8,5] * Conv_1_aggregated_features[5,7]
   + A[9,5] * Conv_1_aggregated_features[5,7] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,8] - A[5,9] * Conv_0_Relu_ts[9,8]
   + A[6,5] * Conv_1_aggregated_features[5,8]
   + A[7,5] * Conv_1_aggregated_features[5,8]
   + A[8,5] * Conv_1_aggregated_features[5,8]
   + A[9,5] * Conv_1_aggregated_features[5,8] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,9] - A[5,9] * Conv_0_Relu_ts[9,9]
   + A[6,5] * Conv_1_aggregated_features[5,9]
   + A[7,5] * Conv_1_aggregated_features[5,9]
   + A[8,5] * Conv_1_aggregated_features[5,9]
   + A[9,5] * Conv_1_aggregated_features[5,9] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,10] - A[5,9] * Conv_0_Relu_ts[9,10]
   + A[6,5] * Conv_1_aggregated_features[5,10]
   + A[7,5] * Conv_1_aggregated_features[5,10]
   + A[8,5] * Conv_1_aggregated_features[5,10]
   + A[9,5] * Conv_1_aggregated_features[5,10] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,11] - A[5,9] * Conv_0_Relu_ts[9,11]
   + A[6,5] * Conv_1_aggregated_features[5,11]
   + A[7,5] * Conv_1_aggregated_features[5,11]
   + A[8,5] * Conv_1_aggregated_features[5,11]
   + A[9,5] * Conv_1_aggregated_features[5,11] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,12] - A[5,9] * Conv_0_Relu_ts[9,12]
   + A[6,5] * Conv_1_aggregated_features[5,12]
   + A[7,5] * Conv_1_aggregated_features[5,12]
   + A[8,5] * Conv_1_aggregated_features[5,12]
   + A[9,5] * Conv_1_aggregated_features[5,12] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,13] - A[5,9] * Conv_0_Relu_ts[9,13]
   + A[6,5] * Conv_1_aggregated_features[5,13]
   + A[7,5] * Conv_1_aggregated_features[5,13]
   + A[8,5] * Conv_1_aggregated_features[5,13]
   + A[9,5] * Conv_1_aggregated_features[5,13] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,14] - A[5,9] * Conv_0_Relu_ts[9,14]
   + A[6,5] * Conv_1_aggregated_features[5,14]
   + A[7,5] * Conv_1_aggregated_features[5,14]
   + A[8,5] * Conv_1_aggregated_features[5,14]
   + A[9,5] * Conv_1_aggregated_features[5,14] ] = 0
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
   - A[5,8] * Conv_0_Relu_ts[8,15] - A[5,9] * Conv_0_Relu_ts[9,15]
   + A[6,5] * Conv_1_aggregated_features[5,15]
   + A[7,5] * Conv_1_aggregated_features[5,15]
   + A[8,5] * Conv_1_aggregated_features[5,15]
   + A[9,5] * Conv_1_aggregated_features[5,15] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,0] - A[6,8] * Conv_0_Relu_ts[8,0]
   - A[6,9] * Conv_0_Relu_ts[9,0]
   + A[7,6] * Conv_1_aggregated_features[6,0]
   + A[8,6] * Conv_1_aggregated_features[6,0]
   + A[9,6] * Conv_1_aggregated_features[6,0] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,1] - A[6,8] * Conv_0_Relu_ts[8,1]
   - A[6,9] * Conv_0_Relu_ts[9,1]
   + A[7,6] * Conv_1_aggregated_features[6,1]
   + A[8,6] * Conv_1_aggregated_features[6,1]
   + A[9,6] * Conv_1_aggregated_features[6,1] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,2] - A[6,8] * Conv_0_Relu_ts[8,2]
   - A[6,9] * Conv_0_Relu_ts[9,2]
   + A[7,6] * Conv_1_aggregated_features[6,2]
   + A[8,6] * Conv_1_aggregated_features[6,2]
   + A[9,6] * Conv_1_aggregated_features[6,2] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,3] - A[6,8] * Conv_0_Relu_ts[8,3]
   - A[6,9] * Conv_0_Relu_ts[9,3]
   + A[7,6] * Conv_1_aggregated_features[6,3]
   + A[8,6] * Conv_1_aggregated_features[6,3]
   + A[9,6] * Conv_1_aggregated_features[6,3] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,4] - A[6,8] * Conv_0_Relu_ts[8,4]
   - A[6,9] * Conv_0_Relu_ts[9,4]
   + A[7,6] * Conv_1_aggregated_features[6,4]
   + A[8,6] * Conv_1_aggregated_features[6,4]
   + A[9,6] * Conv_1_aggregated_features[6,4] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,5] - A[6,8] * Conv_0_Relu_ts[8,5]
   - A[6,9] * Conv_0_Relu_ts[9,5]
   + A[7,6] * Conv_1_aggregated_features[6,5]
   + A[8,6] * Conv_1_aggregated_features[6,5]
   + A[9,6] * Conv_1_aggregated_features[6,5] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,6] - A[6,8] * Conv_0_Relu_ts[8,6]
   - A[6,9] * Conv_0_Relu_ts[9,6]
   + A[7,6] * Conv_1_aggregated_features[6,6]
   + A[8,6] * Conv_1_aggregated_features[6,6]
   + A[9,6] * Conv_1_aggregated_features[6,6] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,7] - A[6,8] * Conv_0_Relu_ts[8,7]
   - A[6,9] * Conv_0_Relu_ts[9,7]
   + A[7,6] * Conv_1_aggregated_features[6,7]
   + A[8,6] * Conv_1_aggregated_features[6,7]
   + A[9,6] * Conv_1_aggregated_features[6,7] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,8] - A[6,8] * Conv_0_Relu_ts[8,8]
   - A[6,9] * Conv_0_Relu_ts[9,8]
   + A[7,6] * Conv_1_aggregated_features[6,8]
   + A[8,6] * Conv_1_aggregated_features[6,8]
   + A[9,6] * Conv_1_aggregated_features[6,8] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,9] - A[6,8] * Conv_0_Relu_ts[8,9]
   - A[6,9] * Conv_0_Relu_ts[9,9]
   + A[7,6] * Conv_1_aggregated_features[6,9]
   + A[8,6] * Conv_1_aggregated_features[6,9]
   + A[9,6] * Conv_1_aggregated_features[6,9] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,10] - A[6,8] * Conv_0_Relu_ts[8,10]
   - A[6,9] * Conv_0_Relu_ts[9,10]
   + A[7,6] * Conv_1_aggregated_features[6,10]
   + A[8,6] * Conv_1_aggregated_features[6,10]
   + A[9,6] * Conv_1_aggregated_features[6,10] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,11] - A[6,8] * Conv_0_Relu_ts[8,11]
   - A[6,9] * Conv_0_Relu_ts[9,11]
   + A[7,6] * Conv_1_aggregated_features[6,11]
   + A[8,6] * Conv_1_aggregated_features[6,11]
   + A[9,6] * Conv_1_aggregated_features[6,11] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,12] - A[6,8] * Conv_0_Relu_ts[8,12]
   - A[6,9] * Conv_0_Relu_ts[9,12]
   + A[7,6] * Conv_1_aggregated_features[6,12]
   + A[8,6] * Conv_1_aggregated_features[6,12]
   + A[9,6] * Conv_1_aggregated_features[6,12] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,13] - A[6,8] * Conv_0_Relu_ts[8,13]
   - A[6,9] * Conv_0_Relu_ts[9,13]
   + A[7,6] * Conv_1_aggregated_features[6,13]
   + A[8,6] * Conv_1_aggregated_features[6,13]
   + A[9,6] * Conv_1_aggregated_features[6,13] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,14] - A[6,8] * Conv_0_Relu_ts[8,14]
   - A[6,9] * Conv_0_Relu_ts[9,14]
   + A[7,6] * Conv_1_aggregated_features[6,14]
   + A[8,6] * Conv_1_aggregated_features[6,14]
   + A[9,6] * Conv_1_aggregated_features[6,14] ] = 0
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
   - A[6,7] * Conv_0_Relu_ts[7,15] - A[6,8] * Conv_0_Relu_ts[8,15]
   - A[6,9] * Conv_0_Relu_ts[9,15]
   + A[7,6] * Conv_1_aggregated_features[6,15]
   + A[8,6] * Conv_1_aggregated_features[6,15]
   + A[9,6] * Conv_1_aggregated_features[6,15] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,0]
   - A[7,8] * Conv_0_Relu_ts[8,0] - A[7,9] * Conv_0_Relu_ts[9,0]
   + A[8,7] * Conv_1_aggregated_features[7,0]
   + A[9,7] * Conv_1_aggregated_features[7,0] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,1]
   - A[7,8] * Conv_0_Relu_ts[8,1] - A[7,9] * Conv_0_Relu_ts[9,1]
   + A[8,7] * Conv_1_aggregated_features[7,1]
   + A[9,7] * Conv_1_aggregated_features[7,1] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,2]
   - A[7,8] * Conv_0_Relu_ts[8,2] - A[7,9] * Conv_0_Relu_ts[9,2]
   + A[8,7] * Conv_1_aggregated_features[7,2]
   + A[9,7] * Conv_1_aggregated_features[7,2] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,3]
   - A[7,8] * Conv_0_Relu_ts[8,3] - A[7,9] * Conv_0_Relu_ts[9,3]
   + A[8,7] * Conv_1_aggregated_features[7,3]
   + A[9,7] * Conv_1_aggregated_features[7,3] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,4]
   - A[7,8] * Conv_0_Relu_ts[8,4] - A[7,9] * Conv_0_Relu_ts[9,4]
   + A[8,7] * Conv_1_aggregated_features[7,4]
   + A[9,7] * Conv_1_aggregated_features[7,4] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,5]
   - A[7,8] * Conv_0_Relu_ts[8,5] - A[7,9] * Conv_0_Relu_ts[9,5]
   + A[8,7] * Conv_1_aggregated_features[7,5]
   + A[9,7] * Conv_1_aggregated_features[7,5] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,6]
   - A[7,8] * Conv_0_Relu_ts[8,6] - A[7,9] * Conv_0_Relu_ts[9,6]
   + A[8,7] * Conv_1_aggregated_features[7,6]
   + A[9,7] * Conv_1_aggregated_features[7,6] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,7]
   - A[7,8] * Conv_0_Relu_ts[8,7] - A[7,9] * Conv_0_Relu_ts[9,7]
   + A[8,7] * Conv_1_aggregated_features[7,7]
   + A[9,7] * Conv_1_aggregated_features[7,7] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,8]
   - A[7,8] * Conv_0_Relu_ts[8,8] - A[7,9] * Conv_0_Relu_ts[9,8]
   + A[8,7] * Conv_1_aggregated_features[7,8]
   + A[9,7] * Conv_1_aggregated_features[7,8] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,9]
   - A[7,8] * Conv_0_Relu_ts[8,9] - A[7,9] * Conv_0_Relu_ts[9,9]
   + A[8,7] * Conv_1_aggregated_features[7,9]
   + A[9,7] * Conv_1_aggregated_features[7,9] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,10]
   - A[7,8] * Conv_0_Relu_ts[8,10] - A[7,9] * Conv_0_Relu_ts[9,10]
   + A[8,7] * Conv_1_aggregated_features[7,10]
   + A[9,7] * Conv_1_aggregated_features[7,10] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,11]
   - A[7,8] * Conv_0_Relu_ts[8,11] - A[7,9] * Conv_0_Relu_ts[9,11]
   + A[8,7] * Conv_1_aggregated_features[7,11]
   + A[9,7] * Conv_1_aggregated_features[7,11] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,12]
   - A[7,8] * Conv_0_Relu_ts[8,12] - A[7,9] * Conv_0_Relu_ts[9,12]
   + A[8,7] * Conv_1_aggregated_features[7,12]
   + A[9,7] * Conv_1_aggregated_features[7,12] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,13]
   - A[7,8] * Conv_0_Relu_ts[8,13] - A[7,9] * Conv_0_Relu_ts[9,13]
   + A[8,7] * Conv_1_aggregated_features[7,13]
   + A[9,7] * Conv_1_aggregated_features[7,13] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,14]
   - A[7,8] * Conv_0_Relu_ts[8,14] - A[7,9] * Conv_0_Relu_ts[9,14]
   + A[8,7] * Conv_1_aggregated_features[7,14]
   + A[9,7] * Conv_1_aggregated_features[7,14] ] = 0
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
   + A[7,7] * Conv_1_aggregated_features[7,15]
   - A[7,8] * Conv_0_Relu_ts[8,15] - A[7,9] * Conv_0_Relu_ts[9,15]
   + A[8,7] * Conv_1_aggregated_features[7,15]
   + A[9,7] * Conv_1_aggregated_features[7,15] ] = 0
 Conv_1_averages_constraint[8,0]: [
   A[0,8] * Conv_1_aggregated_features[8,0]
   + A[1,8] * Conv_1_aggregated_features[8,0]
   + A[2,8] * Conv_1_aggregated_features[8,0]
   + A[3,8] * Conv_1_aggregated_features[8,0]
   + A[4,8] * Conv_1_aggregated_features[8,0]
   + A[5,8] * Conv_1_aggregated_features[8,0]
   + A[6,8] * Conv_1_aggregated_features[8,0]
   + A[7,8] * Conv_1_aggregated_features[8,0]
   - A[8,0] * Conv_0_Relu_ts[0,0] - A[8,1] * Conv_0_Relu_ts[1,0]
   - A[8,2] * Conv_0_Relu_ts[2,0] - A[8,3] * Conv_0_Relu_ts[3,0]
   - A[8,4] * Conv_0_Relu_ts[4,0] - A[8,5] * Conv_0_Relu_ts[5,0]
   - A[8,6] * Conv_0_Relu_ts[6,0] - A[8,7] * Conv_0_Relu_ts[7,0]
   - A[8,8] * Conv_0_Relu_ts[8,0]
   + A[8,8] * Conv_1_aggregated_features[8,0]
   - A[8,9] * Conv_0_Relu_ts[9,0]
   + A[9,8] * Conv_1_aggregated_features[8,0] ] = 0
 Conv_1_averages_constraint[8,1]: [
   A[0,8] * Conv_1_aggregated_features[8,1]
   + A[1,8] * Conv_1_aggregated_features[8,1]
   + A[2,8] * Conv_1_aggregated_features[8,1]
   + A[3,8] * Conv_1_aggregated_features[8,1]
   + A[4,8] * Conv_1_aggregated_features[8,1]
   + A[5,8] * Conv_1_aggregated_features[8,1]
   + A[6,8] * Conv_1_aggregated_features[8,1]
   + A[7,8] * Conv_1_aggregated_features[8,1]
   - A[8,0] * Conv_0_Relu_ts[0,1] - A[8,1] * Conv_0_Relu_ts[1,1]
   - A[8,2] * Conv_0_Relu_ts[2,1] - A[8,3] * Conv_0_Relu_ts[3,1]
   - A[8,4] * Conv_0_Relu_ts[4,1] - A[8,5] * Conv_0_Relu_ts[5,1]
   - A[8,6] * Conv_0_Relu_ts[6,1] - A[8,7] * Conv_0_Relu_ts[7,1]
   - A[8,8] * Conv_0_Relu_ts[8,1]
   + A[8,8] * Conv_1_aggregated_features[8,1]
   - A[8,9] * Conv_0_Relu_ts[9,1]
   + A[9,8] * Conv_1_aggregated_features[8,1] ] = 0
 Conv_1_averages_constraint[8,2]: [
   A[0,8] * Conv_1_aggregated_features[8,2]
   + A[1,8] * Conv_1_aggregated_features[8,2]
   + A[2,8] * Conv_1_aggregated_features[8,2]
   + A[3,8] * Conv_1_aggregated_features[8,2]
   + A[4,8] * Conv_1_aggregated_features[8,2]
   + A[5,8] * Conv_1_aggregated_features[8,2]
   + A[6,8] * Conv_1_aggregated_features[8,2]
   + A[7,8] * Conv_1_aggregated_features[8,2]
   - A[8,0] * Conv_0_Relu_ts[0,2] - A[8,1] * Conv_0_Relu_ts[1,2]
   - A[8,2] * Conv_0_Relu_ts[2,2] - A[8,3] * Conv_0_Relu_ts[3,2]
   - A[8,4] * Conv_0_Relu_ts[4,2] - A[8,5] * Conv_0_Relu_ts[5,2]
   - A[8,6] * Conv_0_Relu_ts[6,2] - A[8,7] * Conv_0_Relu_ts[7,2]
   - A[8,8] * Conv_0_Relu_ts[8,2]
   + A[8,8] * Conv_1_aggregated_features[8,2]
   - A[8,9] * Conv_0_Relu_ts[9,2]
   + A[9,8] * Conv_1_aggregated_features[8,2] ] = 0
 Conv_1_averages_constraint[8,3]: [
   A[0,8] * Conv_1_aggregated_features[8,3]
   + A[1,8] * Conv_1_aggregated_features[8,3]
   + A[2,8] * Conv_1_aggregated_features[8,3]
   + A[3,8] * Conv_1_aggregated_features[8,3]
   + A[4,8] * Conv_1_aggregated_features[8,3]
   + A[5,8] * Conv_1_aggregated_features[8,3]
   + A[6,8] * Conv_1_aggregated_features[8,3]
   + A[7,8] * Conv_1_aggregated_features[8,3]
   - A[8,0] * Conv_0_Relu_ts[0,3] - A[8,1] * Conv_0_Relu_ts[1,3]
   - A[8,2] * Conv_0_Relu_ts[2,3] - A[8,3] * Conv_0_Relu_ts[3,3]
   - A[8,4] * Conv_0_Relu_ts[4,3] - A[8,5] * Conv_0_Relu_ts[5,3]
   - A[8,6] * Conv_0_Relu_ts[6,3] - A[8,7] * Conv_0_Relu_ts[7,3]
   - A[8,8] * Conv_0_Relu_ts[8,3]
   + A[8,8] * Conv_1_aggregated_features[8,3]
   - A[8,9] * Conv_0_Relu_ts[9,3]
   + A[9,8] * Conv_1_aggregated_features[8,3] ] = 0
 Conv_1_averages_constraint[8,4]: [
   A[0,8] * Conv_1_aggregated_features[8,4]
   + A[1,8] * Conv_1_aggregated_features[8,4]
   + A[2,8] * Conv_1_aggregated_features[8,4]
   + A[3,8] * Conv_1_aggregated_features[8,4]
   + A[4,8] * Conv_1_aggregated_features[8,4]
   + A[5,8] * Conv_1_aggregated_features[8,4]
   + A[6,8] * Conv_1_aggregated_features[8,4]
   + A[7,8] * Conv_1_aggregated_features[8,4]
   - A[8,0] * Conv_0_Relu_ts[0,4] - A[8,1] * Conv_0_Relu_ts[1,4]
   - A[8,2] * Conv_0_Relu_ts[2,4] - A[8,3] * Conv_0_Relu_ts[3,4]
   - A[8,4] * Conv_0_Relu_ts[4,4] - A[8,5] * Conv_0_Relu_ts[5,4]
   - A[8,6] * Conv_0_Relu_ts[6,4] - A[8,7] * Conv_0_Relu_ts[7,4]
   - A[8,8] * Conv_0_Relu_ts[8,4]
   + A[8,8] * Conv_1_aggregated_features[8,4]
   - A[8,9] * Conv_0_Relu_ts[9,4]
   + A[9,8] * Conv_1_aggregated_features[8,4] ] = 0
 Conv_1_averages_constraint[8,5]: [
   A[0,8] * Conv_1_aggregated_features[8,5]
   + A[1,8] * Conv_1_aggregated_features[8,5]
   + A[2,8] * Conv_1_aggregated_features[8,5]
   + A[3,8] * Conv_1_aggregated_features[8,5]
   + A[4,8] * Conv_1_aggregated_features[8,5]
   + A[5,8] * Conv_1_aggregated_features[8,5]
   + A[6,8] * Conv_1_aggregated_features[8,5]
   + A[7,8] * Conv_1_aggregated_features[8,5]
   - A[8,0] * Conv_0_Relu_ts[0,5] - A[8,1] * Conv_0_Relu_ts[1,5]
   - A[8,2] * Conv_0_Relu_ts[2,5] - A[8,3] * Conv_0_Relu_ts[3,5]
   - A[8,4] * Conv_0_Relu_ts[4,5] - A[8,5] * Conv_0_Relu_ts[5,5]
   - A[8,6] * Conv_0_Relu_ts[6,5] - A[8,7] * Conv_0_Relu_ts[7,5]
   - A[8,8] * Conv_0_Relu_ts[8,5]
   + A[8,8] * Conv_1_aggregated_features[8,5]
   - A[8,9] * Conv_0_Relu_ts[9,5]
   + A[9,8] * Conv_1_aggregated_features[8,5] ] = 0
 Conv_1_averages_constraint[8,6]: [
   A[0,8] * Conv_1_aggregated_features[8,6]
   + A[1,8] * Conv_1_aggregated_features[8,6]
   + A[2,8] * Conv_1_aggregated_features[8,6]
   + A[3,8] * Conv_1_aggregated_features[8,6]
   + A[4,8] * Conv_1_aggregated_features[8,6]
   + A[5,8] * Conv_1_aggregated_features[8,6]
   + A[6,8] * Conv_1_aggregated_features[8,6]
   + A[7,8] * Conv_1_aggregated_features[8,6]
   - A[8,0] * Conv_0_Relu_ts[0,6] - A[8,1] * Conv_0_Relu_ts[1,6]
   - A[8,2] * Conv_0_Relu_ts[2,6] - A[8,3] * Conv_0_Relu_ts[3,6]
   - A[8,4] * Conv_0_Relu_ts[4,6] - A[8,5] * Conv_0_Relu_ts[5,6]
   - A[8,6] * Conv_0_Relu_ts[6,6] - A[8,7] * Conv_0_Relu_ts[7,6]
   - A[8,8] * Conv_0_Relu_ts[8,6]
   + A[8,8] * Conv_1_aggregated_features[8,6]
   - A[8,9] * Conv_0_Relu_ts[9,6]
   + A[9,8] * Conv_1_aggregated_features[8,6] ] = 0
 Conv_1_averages_constraint[8,7]: [
   A[0,8] * Conv_1_aggregated_features[8,7]
   + A[1,8] * Conv_1_aggregated_features[8,7]
   + A[2,8] * Conv_1_aggregated_features[8,7]
   + A[3,8] * Conv_1_aggregated_features[8,7]
   + A[4,8] * Conv_1_aggregated_features[8,7]
   + A[5,8] * Conv_1_aggregated_features[8,7]
   + A[6,8] * Conv_1_aggregated_features[8,7]
   + A[7,8] * Conv_1_aggregated_features[8,7]
   - A[8,0] * Conv_0_Relu_ts[0,7] - A[8,1] * Conv_0_Relu_ts[1,7]
   - A[8,2] * Conv_0_Relu_ts[2,7] - A[8,3] * Conv_0_Relu_ts[3,7]
   - A[8,4] * Conv_0_Relu_ts[4,7] - A[8,5] * Conv_0_Relu_ts[5,7]
   - A[8,6] * Conv_0_Relu_ts[6,7] - A[8,7] * Conv_0_Relu_ts[7,7]
   - A[8,8] * Conv_0_Relu_ts[8,7]
   + A[8,8] * Conv_1_aggregated_features[8,7]
   - A[8,9] * Conv_0_Relu_ts[9,7]
   + A[9,8] * Conv_1_aggregated_features[8,7] ] = 0
 Conv_1_averages_constraint[8,8]: [
   A[0,8] * Conv_1_aggregated_features[8,8]
   + A[1,8] * Conv_1_aggregated_features[8,8]
   + A[2,8] * Conv_1_aggregated_features[8,8]
   + A[3,8] * Conv_1_aggregated_features[8,8]
   + A[4,8] * Conv_1_aggregated_features[8,8]
   + A[5,8] * Conv_1_aggregated_features[8,8]
   + A[6,8] * Conv_1_aggregated_features[8,8]
   + A[7,8] * Conv_1_aggregated_features[8,8]
   - A[8,0] * Conv_0_Relu_ts[0,8] - A[8,1] * Conv_0_Relu_ts[1,8]
   - A[8,2] * Conv_0_Relu_ts[2,8] - A[8,3] * Conv_0_Relu_ts[3,8]
   - A[8,4] * Conv_0_Relu_ts[4,8] - A[8,5] * Conv_0_Relu_ts[5,8]
   - A[8,6] * Conv_0_Relu_ts[6,8] - A[8,7] * Conv_0_Relu_ts[7,8]
   - A[8,8] * Conv_0_Relu_ts[8,8]
   + A[8,8] * Conv_1_aggregated_features[8,8]
   - A[8,9] * Conv_0_Relu_ts[9,8]
   + A[9,8] * Conv_1_aggregated_features[8,8] ] = 0
 Conv_1_averages_constraint[8,9]: [
   A[0,8] * Conv_1_aggregated_features[8,9]
   + A[1,8] * Conv_1_aggregated_features[8,9]
   + A[2,8] * Conv_1_aggregated_features[8,9]
   + A[3,8] * Conv_1_aggregated_features[8,9]
   + A[4,8] * Conv_1_aggregated_features[8,9]
   + A[5,8] * Conv_1_aggregated_features[8,9]
   + A[6,8] * Conv_1_aggregated_features[8,9]
   + A[7,8] * Conv_1_aggregated_features[8,9]
   - A[8,0] * Conv_0_Relu_ts[0,9] - A[8,1] * Conv_0_Relu_ts[1,9]
   - A[8,2] * Conv_0_Relu_ts[2,9] - A[8,3] * Conv_0_Relu_ts[3,9]
   - A[8,4] * Conv_0_Relu_ts[4,9] - A[8,5] * Conv_0_Relu_ts[5,9]
   - A[8,6] * Conv_0_Relu_ts[6,9] - A[8,7] * Conv_0_Relu_ts[7,9]
   - A[8,8] * Conv_0_Relu_ts[8,9]
   + A[8,8] * Conv_1_aggregated_features[8,9]
   - A[8,9] * Conv_0_Relu_ts[9,9]
   + A[9,8] * Conv_1_aggregated_features[8,9] ] = 0
 Conv_1_averages_constraint[8,10]: [
   A[0,8] * Conv_1_aggregated_features[8,10]
   + A[1,8] * Conv_1_aggregated_features[8,10]
   + A[2,8] * Conv_1_aggregated_features[8,10]
   + A[3,8] * Conv_1_aggregated_features[8,10]
   + A[4,8] * Conv_1_aggregated_features[8,10]
   + A[5,8] * Conv_1_aggregated_features[8,10]
   + A[6,8] * Conv_1_aggregated_features[8,10]
   + A[7,8] * Conv_1_aggregated_features[8,10]
   - A[8,0] * Conv_0_Relu_ts[0,10] - A[8,1] * Conv_0_Relu_ts[1,10]
   - A[8,2] * Conv_0_Relu_ts[2,10] - A[8,3] * Conv_0_Relu_ts[3,10]
   - A[8,4] * Conv_0_Relu_ts[4,10] - A[8,5] * Conv_0_Relu_ts[5,10]
   - A[8,6] * Conv_0_Relu_ts[6,10] - A[8,7] * Conv_0_Relu_ts[7,10]
   - A[8,8] * Conv_0_Relu_ts[8,10]
   + A[8,8] * Conv_1_aggregated_features[8,10]
   - A[8,9] * Conv_0_Relu_ts[9,10]
   + A[9,8] * Conv_1_aggregated_features[8,10] ] = 0
 Conv_1_averages_constraint[8,11]: [
   A[0,8] * Conv_1_aggregated_features[8,11]
   + A[1,8] * Conv_1_aggregated_features[8,11]
   + A[2,8] * Conv_1_aggregated_features[8,11]
   + A[3,8] * Conv_1_aggregated_features[8,11]
   + A[4,8] * Conv_1_aggregated_features[8,11]
   + A[5,8] * Conv_1_aggregated_features[8,11]
   + A[6,8] * Conv_1_aggregated_features[8,11]
   + A[7,8] * Conv_1_aggregated_features[8,11]
   - A[8,0] * Conv_0_Relu_ts[0,11] - A[8,1] * Conv_0_Relu_ts[1,11]
   - A[8,2] * Conv_0_Relu_ts[2,11] - A[8,3] * Conv_0_Relu_ts[3,11]
   - A[8,4] * Conv_0_Relu_ts[4,11] - A[8,5] * Conv_0_Relu_ts[5,11]
   - A[8,6] * Conv_0_Relu_ts[6,11] - A[8,7] * Conv_0_Relu_ts[7,11]
   - A[8,8] * Conv_0_Relu_ts[8,11]
   + A[8,8] * Conv_1_aggregated_features[8,11]
   - A[8,9] * Conv_0_Relu_ts[9,11]
   + A[9,8] * Conv_1_aggregated_features[8,11] ] = 0
 Conv_1_averages_constraint[8,12]: [
   A[0,8] * Conv_1_aggregated_features[8,12]
   + A[1,8] * Conv_1_aggregated_features[8,12]
   + A[2,8] * Conv_1_aggregated_features[8,12]
   + A[3,8] * Conv_1_aggregated_features[8,12]
   + A[4,8] * Conv_1_aggregated_features[8,12]
   + A[5,8] * Conv_1_aggregated_features[8,12]
   + A[6,8] * Conv_1_aggregated_features[8,12]
   + A[7,8] * Conv_1_aggregated_features[8,12]
   - A[8,0] * Conv_0_Relu_ts[0,12] - A[8,1] * Conv_0_Relu_ts[1,12]
   - A[8,2] * Conv_0_Relu_ts[2,12] - A[8,3] * Conv_0_Relu_ts[3,12]
   - A[8,4] * Conv_0_Relu_ts[4,12] - A[8,5] * Conv_0_Relu_ts[5,12]
   - A[8,6] * Conv_0_Relu_ts[6,12] - A[8,7] * Conv_0_Relu_ts[7,12]
   - A[8,8] * Conv_0_Relu_ts[8,12]
   + A[8,8] * Conv_1_aggregated_features[8,12]
   - A[8,9] * Conv_0_Relu_ts[9,12]
   + A[9,8] * Conv_1_aggregated_features[8,12] ] = 0
 Conv_1_averages_constraint[8,13]: [
   A[0,8] * Conv_1_aggregated_features[8,13]
   + A[1,8] * Conv_1_aggregated_features[8,13]
   + A[2,8] * Conv_1_aggregated_features[8,13]
   + A[3,8] * Conv_1_aggregated_features[8,13]
   + A[4,8] * Conv_1_aggregated_features[8,13]
   + A[5,8] * Conv_1_aggregated_features[8,13]
   + A[6,8] * Conv_1_aggregated_features[8,13]
   + A[7,8] * Conv_1_aggregated_features[8,13]
   - A[8,0] * Conv_0_Relu_ts[0,13] - A[8,1] * Conv_0_Relu_ts[1,13]
   - A[8,2] * Conv_0_Relu_ts[2,13] - A[8,3] * Conv_0_Relu_ts[3,13]
   - A[8,4] * Conv_0_Relu_ts[4,13] - A[8,5] * Conv_0_Relu_ts[5,13]
   - A[8,6] * Conv_0_Relu_ts[6,13] - A[8,7] * Conv_0_Relu_ts[7,13]
   - A[8,8] * Conv_0_Relu_ts[8,13]
   + A[8,8] * Conv_1_aggregated_features[8,13]
   - A[8,9] * Conv_0_Relu_ts[9,13]
   + A[9,8] * Conv_1_aggregated_features[8,13] ] = 0
 Conv_1_averages_constraint[8,14]: [
   A[0,8] * Conv_1_aggregated_features[8,14]
   + A[1,8] * Conv_1_aggregated_features[8,14]
   + A[2,8] * Conv_1_aggregated_features[8,14]
   + A[3,8] * Conv_1_aggregated_features[8,14]
   + A[4,8] * Conv_1_aggregated_features[8,14]
   + A[5,8] * Conv_1_aggregated_features[8,14]
   + A[6,8] * Conv_1_aggregated_features[8,14]
   + A[7,8] * Conv_1_aggregated_features[8,14]
   - A[8,0] * Conv_0_Relu_ts[0,14] - A[8,1] * Conv_0_Relu_ts[1,14]
   - A[8,2] * Conv_0_Relu_ts[2,14] - A[8,3] * Conv_0_Relu_ts[3,14]
   - A[8,4] * Conv_0_Relu_ts[4,14] - A[8,5] * Conv_0_Relu_ts[5,14]
   - A[8,6] * Conv_0_Relu_ts[6,14] - A[8,7] * Conv_0_Relu_ts[7,14]
   - A[8,8] * Conv_0_Relu_ts[8,14]
   + A[8,8] * Conv_1_aggregated_features[8,14]
   - A[8,9] * Conv_0_Relu_ts[9,14]
   + A[9,8] * Conv_1_aggregated_features[8,14] ] = 0
 Conv_1_averages_constraint[8,15]: [
   A[0,8] * Conv_1_aggregated_features[8,15]
   + A[1,8] * Conv_1_aggregated_features[8,15]
   + A[2,8] * Conv_1_aggregated_features[8,15]
   + A[3,8] * Conv_1_aggregated_features[8,15]
   + A[4,8] * Conv_1_aggregated_features[8,15]
   + A[5,8] * Conv_1_aggregated_features[8,15]
   + A[6,8] * Conv_1_aggregated_features[8,15]
   + A[7,8] * Conv_1_aggregated_features[8,15]
   - A[8,0] * Conv_0_Relu_ts[0,15] - A[8,1] * Conv_0_Relu_ts[1,15]
   - A[8,2] * Conv_0_Relu_ts[2,15] - A[8,3] * Conv_0_Relu_ts[3,15]
   - A[8,4] * Conv_0_Relu_ts[4,15] - A[8,5] * Conv_0_Relu_ts[5,15]
   - A[8,6] * Conv_0_Relu_ts[6,15] - A[8,7] * Conv_0_Relu_ts[7,15]
   - A[8,8] * Conv_0_Relu_ts[8,15]
   + A[8,8] * Conv_1_aggregated_features[8,15]
   - A[8,9] * Conv_0_Relu_ts[9,15]
   + A[9,8] * Conv_1_aggregated_features[8,15] ] = 0
 Conv_1_averages_constraint[9,0]: [
   A[0,9] * Conv_1_aggregated_features[9,0]
   + A[1,9] * Conv_1_aggregated_features[9,0]
   + A[2,9] * Conv_1_aggregated_features[9,0]
   + A[3,9] * Conv_1_aggregated_features[9,0]
   + A[4,9] * Conv_1_aggregated_features[9,0]
   + A[5,9] * Conv_1_aggregated_features[9,0]
   + A[6,9] * Conv_1_aggregated_features[9,0]
   + A[7,9] * Conv_1_aggregated_features[9,0]
   + A[8,9] * Conv_1_aggregated_features[9,0]
   - A[9,0] * Conv_0_Relu_ts[0,0] - A[9,1] * Conv_0_Relu_ts[1,0]
   - A[9,2] * Conv_0_Relu_ts[2,0] - A[9,3] * Conv_0_Relu_ts[3,0]
   - A[9,4] * Conv_0_Relu_ts[4,0] - A[9,5] * Conv_0_Relu_ts[5,0]
   - A[9,6] * Conv_0_Relu_ts[6,0] - A[9,7] * Conv_0_Relu_ts[7,0]
   - A[9,8] * Conv_0_Relu_ts[8,0] - A[9,9] * Conv_0_Relu_ts[9,0]
   + A[9,9] * Conv_1_aggregated_features[9,0] ] = 0
 Conv_1_averages_constraint[9,1]: [
   A[0,9] * Conv_1_aggregated_features[9,1]
   + A[1,9] * Conv_1_aggregated_features[9,1]
   + A[2,9] * Conv_1_aggregated_features[9,1]
   + A[3,9] * Conv_1_aggregated_features[9,1]
   + A[4,9] * Conv_1_aggregated_features[9,1]
   + A[5,9] * Conv_1_aggregated_features[9,1]
   + A[6,9] * Conv_1_aggregated_features[9,1]
   + A[7,9] * Conv_1_aggregated_features[9,1]
   + A[8,9] * Conv_1_aggregated_features[9,1]
   - A[9,0] * Conv_0_Relu_ts[0,1] - A[9,1] * Conv_0_Relu_ts[1,1]
   - A[9,2] * Conv_0_Relu_ts[2,1] - A[9,3] * Conv_0_Relu_ts[3,1]
   - A[9,4] * Conv_0_Relu_ts[4,1] - A[9,5] * Conv_0_Relu_ts[5,1]
   - A[9,6] * Conv_0_Relu_ts[6,1] - A[9,7] * Conv_0_Relu_ts[7,1]
   - A[9,8] * Conv_0_Relu_ts[8,1] - A[9,9] * Conv_0_Relu_ts[9,1]
   + A[9,9] * Conv_1_aggregated_features[9,1] ] = 0
 Conv_1_averages_constraint[9,2]: [
   A[0,9] * Conv_1_aggregated_features[9,2]
   + A[1,9] * Conv_1_aggregated_features[9,2]
   + A[2,9] * Conv_1_aggregated_features[9,2]
   + A[3,9] * Conv_1_aggregated_features[9,2]
   + A[4,9] * Conv_1_aggregated_features[9,2]
   + A[5,9] * Conv_1_aggregated_features[9,2]
   + A[6,9] * Conv_1_aggregated_features[9,2]
   + A[7,9] * Conv_1_aggregated_features[9,2]
   + A[8,9] * Conv_1_aggregated_features[9,2]
   - A[9,0] * Conv_0_Relu_ts[0,2] - A[9,1] * Conv_0_Relu_ts[1,2]
   - A[9,2] * Conv_0_Relu_ts[2,2] - A[9,3] * Conv_0_Relu_ts[3,2]
   - A[9,4] * Conv_0_Relu_ts[4,2] - A[9,5] * Conv_0_Relu_ts[5,2]
   - A[9,6] * Conv_0_Relu_ts[6,2] - A[9,7] * Conv_0_Relu_ts[7,2]
   - A[9,8] * Conv_0_Relu_ts[8,2] - A[9,9] * Conv_0_Relu_ts[9,2]
   + A[9,9] * Conv_1_aggregated_features[9,2] ] = 0
 Conv_1_averages_constraint[9,3]: [
   A[0,9] * Conv_1_aggregated_features[9,3]
   + A[1,9] * Conv_1_aggregated_features[9,3]
   + A[2,9] * Conv_1_aggregated_features[9,3]
   + A[3,9] * Conv_1_aggregated_features[9,3]
   + A[4,9] * Conv_1_aggregated_features[9,3]
   + A[5,9] * Conv_1_aggregated_features[9,3]
   + A[6,9] * Conv_1_aggregated_features[9,3]
   + A[7,9] * Conv_1_aggregated_features[9,3]
   + A[8,9] * Conv_1_aggregated_features[9,3]
   - A[9,0] * Conv_0_Relu_ts[0,3] - A[9,1] * Conv_0_Relu_ts[1,3]
   - A[9,2] * Conv_0_Relu_ts[2,3] - A[9,3] * Conv_0_Relu_ts[3,3]
   - A[9,4] * Conv_0_Relu_ts[4,3] - A[9,5] * Conv_0_Relu_ts[5,3]
   - A[9,6] * Conv_0_Relu_ts[6,3] - A[9,7] * Conv_0_Relu_ts[7,3]
   - A[9,8] * Conv_0_Relu_ts[8,3] - A[9,9] * Conv_0_Relu_ts[9,3]
   + A[9,9] * Conv_1_aggregated_features[9,3] ] = 0
 Conv_1_averages_constraint[9,4]: [
   A[0,9] * Conv_1_aggregated_features[9,4]
   + A[1,9] * Conv_1_aggregated_features[9,4]
   + A[2,9] * Conv_1_aggregated_features[9,4]
   + A[3,9] * Conv_1_aggregated_features[9,4]
   + A[4,9] * Conv_1_aggregated_features[9,4]
   + A[5,9] * Conv_1_aggregated_features[9,4]
   + A[6,9] * Conv_1_aggregated_features[9,4]
   + A[7,9] * Conv_1_aggregated_features[9,4]
   + A[8,9] * Conv_1_aggregated_features[9,4]
   - A[9,0] * Conv_0_Relu_ts[0,4] - A[9,1] * Conv_0_Relu_ts[1,4]
   - A[9,2] * Conv_0_Relu_ts[2,4] - A[9,3] * Conv_0_Relu_ts[3,4]
   - A[9,4] * Conv_0_Relu_ts[4,4] - A[9,5] * Conv_0_Relu_ts[5,4]
   - A[9,6] * Conv_0_Relu_ts[6,4] - A[9,7] * Conv_0_Relu_ts[7,4]
   - A[9,8] * Conv_0_Relu_ts[8,4] - A[9,9] * Conv_0_Relu_ts[9,4]
   + A[9,9] * Conv_1_aggregated_features[9,4] ] = 0
 Conv_1_averages_constraint[9,5]: [
   A[0,9] * Conv_1_aggregated_features[9,5]
   + A[1,9] * Conv_1_aggregated_features[9,5]
   + A[2,9] * Conv_1_aggregated_features[9,5]
   + A[3,9] * Conv_1_aggregated_features[9,5]
   + A[4,9] * Conv_1_aggregated_features[9,5]
   + A[5,9] * Conv_1_aggregated_features[9,5]
   + A[6,9] * Conv_1_aggregated_features[9,5]
   + A[7,9] * Conv_1_aggregated_features[9,5]
   + A[8,9] * Conv_1_aggregated_features[9,5]
   - A[9,0] * Conv_0_Relu_ts[0,5] - A[9,1] * Conv_0_Relu_ts[1,5]
   - A[9,2] * Conv_0_Relu_ts[2,5] - A[9,3] * Conv_0_Relu_ts[3,5]
   - A[9,4] * Conv_0_Relu_ts[4,5] - A[9,5] * Conv_0_Relu_ts[5,5]
   - A[9,6] * Conv_0_Relu_ts[6,5] - A[9,7] * Conv_0_Relu_ts[7,5]
   - A[9,8] * Conv_0_Relu_ts[8,5] - A[9,9] * Conv_0_Relu_ts[9,5]
   + A[9,9] * Conv_1_aggregated_features[9,5] ] = 0
 Conv_1_averages_constraint[9,6]: [
   A[0,9] * Conv_1_aggregated_features[9,6]
   + A[1,9] * Conv_1_aggregated_features[9,6]
   + A[2,9] * Conv_1_aggregated_features[9,6]
   + A[3,9] * Conv_1_aggregated_features[9,6]
   + A[4,9] * Conv_1_aggregated_features[9,6]
   + A[5,9] * Conv_1_aggregated_features[9,6]
   + A[6,9] * Conv_1_aggregated_features[9,6]
   + A[7,9] * Conv_1_aggregated_features[9,6]
   + A[8,9] * Conv_1_aggregated_features[9,6]
   - A[9,0] * Conv_0_Relu_ts[0,6] - A[9,1] * Conv_0_Relu_ts[1,6]
   - A[9,2] * Conv_0_Relu_ts[2,6] - A[9,3] * Conv_0_Relu_ts[3,6]
   - A[9,4] * Conv_0_Relu_ts[4,6] - A[9,5] * Conv_0_Relu_ts[5,6]
   - A[9,6] * Conv_0_Relu_ts[6,6] - A[9,7] * Conv_0_Relu_ts[7,6]
   - A[9,8] * Conv_0_Relu_ts[8,6] - A[9,9] * Conv_0_Relu_ts[9,6]
   + A[9,9] * Conv_1_aggregated_features[9,6] ] = 0
 Conv_1_averages_constraint[9,7]: [
   A[0,9] * Conv_1_aggregated_features[9,7]
   + A[1,9] * Conv_1_aggregated_features[9,7]
   + A[2,9] * Conv_1_aggregated_features[9,7]
   + A[3,9] * Conv_1_aggregated_features[9,7]
   + A[4,9] * Conv_1_aggregated_features[9,7]
   + A[5,9] * Conv_1_aggregated_features[9,7]
   + A[6,9] * Conv_1_aggregated_features[9,7]
   + A[7,9] * Conv_1_aggregated_features[9,7]
   + A[8,9] * Conv_1_aggregated_features[9,7]
   - A[9,0] * Conv_0_Relu_ts[0,7] - A[9,1] * Conv_0_Relu_ts[1,7]
   - A[9,2] * Conv_0_Relu_ts[2,7] - A[9,3] * Conv_0_Relu_ts[3,7]
   - A[9,4] * Conv_0_Relu_ts[4,7] - A[9,5] * Conv_0_Relu_ts[5,7]
   - A[9,6] * Conv_0_Relu_ts[6,7] - A[9,7] * Conv_0_Relu_ts[7,7]
   - A[9,8] * Conv_0_Relu_ts[8,7] - A[9,9] * Conv_0_Relu_ts[9,7]
   + A[9,9] * Conv_1_aggregated_features[9,7] ] = 0
 Conv_1_averages_constraint[9,8]: [
   A[0,9] * Conv_1_aggregated_features[9,8]
   + A[1,9] * Conv_1_aggregated_features[9,8]
   + A[2,9] * Conv_1_aggregated_features[9,8]
   + A[3,9] * Conv_1_aggregated_features[9,8]
   + A[4,9] * Conv_1_aggregated_features[9,8]
   + A[5,9] * Conv_1_aggregated_features[9,8]
   + A[6,9] * Conv_1_aggregated_features[9,8]
   + A[7,9] * Conv_1_aggregated_features[9,8]
   + A[8,9] * Conv_1_aggregated_features[9,8]
   - A[9,0] * Conv_0_Relu_ts[0,8] - A[9,1] * Conv_0_Relu_ts[1,8]
   - A[9,2] * Conv_0_Relu_ts[2,8] - A[9,3] * Conv_0_Relu_ts[3,8]
   - A[9,4] * Conv_0_Relu_ts[4,8] - A[9,5] * Conv_0_Relu_ts[5,8]
   - A[9,6] * Conv_0_Relu_ts[6,8] - A[9,7] * Conv_0_Relu_ts[7,8]
   - A[9,8] * Conv_0_Relu_ts[8,8] - A[9,9] * Conv_0_Relu_ts[9,8]
   + A[9,9] * Conv_1_aggregated_features[9,8] ] = 0
 Conv_1_averages_constraint[9,9]: [
   A[0,9] * Conv_1_aggregated_features[9,9]
   + A[1,9] * Conv_1_aggregated_features[9,9]
   + A[2,9] * Conv_1_aggregated_features[9,9]
   + A[3,9] * Conv_1_aggregated_features[9,9]
   + A[4,9] * Conv_1_aggregated_features[9,9]
   + A[5,9] * Conv_1_aggregated_features[9,9]
   + A[6,9] * Conv_1_aggregated_features[9,9]
   + A[7,9] * Conv_1_aggregated_features[9,9]
   + A[8,9] * Conv_1_aggregated_features[9,9]
   - A[9,0] * Conv_0_Relu_ts[0,9] - A[9,1] * Conv_0_Relu_ts[1,9]
   - A[9,2] * Conv_0_Relu_ts[2,9] - A[9,3] * Conv_0_Relu_ts[3,9]
   - A[9,4] * Conv_0_Relu_ts[4,9] - A[9,5] * Conv_0_Relu_ts[5,9]
   - A[9,6] * Conv_0_Relu_ts[6,9] - A[9,7] * Conv_0_Relu_ts[7,9]
   - A[9,8] * Conv_0_Relu_ts[8,9] - A[9,9] * Conv_0_Relu_ts[9,9]
   + A[9,9] * Conv_1_aggregated_features[9,9] ] = 0
 Conv_1_averages_constraint[9,10]: [
   A[0,9] * Conv_1_aggregated_features[9,10]
   + A[1,9] * Conv_1_aggregated_features[9,10]
   + A[2,9] * Conv_1_aggregated_features[9,10]
   + A[3,9] * Conv_1_aggregated_features[9,10]
   + A[4,9] * Conv_1_aggregated_features[9,10]
   + A[5,9] * Conv_1_aggregated_features[9,10]
   + A[6,9] * Conv_1_aggregated_features[9,10]
   + A[7,9] * Conv_1_aggregated_features[9,10]
   + A[8,9] * Conv_1_aggregated_features[9,10]
   - A[9,0] * Conv_0_Relu_ts[0,10] - A[9,1] * Conv_0_Relu_ts[1,10]
   - A[9,2] * Conv_0_Relu_ts[2,10] - A[9,3] * Conv_0_Relu_ts[3,10]
   - A[9,4] * Conv_0_Relu_ts[4,10] - A[9,5] * Conv_0_Relu_ts[5,10]
   - A[9,6] * Conv_0_Relu_ts[6,10] - A[9,7] * Conv_0_Relu_ts[7,10]
   - A[9,8] * Conv_0_Relu_ts[8,10] - A[9,9] * Conv_0_Relu_ts[9,10]
   + A[9,9] * Conv_1_aggregated_features[9,10] ] = 0
 Conv_1_averages_constraint[9,11]: [
   A[0,9] * Conv_1_aggregated_features[9,11]
   + A[1,9] * Conv_1_aggregated_features[9,11]
   + A[2,9] * Conv_1_aggregated_features[9,11]
   + A[3,9] * Conv_1_aggregated_features[9,11]
   + A[4,9] * Conv_1_aggregated_features[9,11]
   + A[5,9] * Conv_1_aggregated_features[9,11]
   + A[6,9] * Conv_1_aggregated_features[9,11]
   + A[7,9] * Conv_1_aggregated_features[9,11]
   + A[8,9] * Conv_1_aggregated_features[9,11]
   - A[9,0] * Conv_0_Relu_ts[0,11] - A[9,1] * Conv_0_Relu_ts[1,11]
   - A[9,2] * Conv_0_Relu_ts[2,11] - A[9,3] * Conv_0_Relu_ts[3,11]
   - A[9,4] * Conv_0_Relu_ts[4,11] - A[9,5] * Conv_0_Relu_ts[5,11]
   - A[9,6] * Conv_0_Relu_ts[6,11] - A[9,7] * Conv_0_Relu_ts[7,11]
   - A[9,8] * Conv_0_Relu_ts[8,11] - A[9,9] * Conv_0_Relu_ts[9,11]
   + A[9,9] * Conv_1_aggregated_features[9,11] ] = 0
 Conv_1_averages_constraint[9,12]: [
   A[0,9] * Conv_1_aggregated_features[9,12]
   + A[1,9] * Conv_1_aggregated_features[9,12]
   + A[2,9] * Conv_1_aggregated_features[9,12]
   + A[3,9] * Conv_1_aggregated_features[9,12]
   + A[4,9] * Conv_1_aggregated_features[9,12]
   + A[5,9] * Conv_1_aggregated_features[9,12]
   + A[6,9] * Conv_1_aggregated_features[9,12]
   + A[7,9] * Conv_1_aggregated_features[9,12]
   + A[8,9] * Conv_1_aggregated_features[9,12]
   - A[9,0] * Conv_0_Relu_ts[0,12] - A[9,1] * Conv_0_Relu_ts[1,12]
   - A[9,2] * Conv_0_Relu_ts[2,12] - A[9,3] * Conv_0_Relu_ts[3,12]
   - A[9,4] * Conv_0_Relu_ts[4,12] - A[9,5] * Conv_0_Relu_ts[5,12]
   - A[9,6] * Conv_0_Relu_ts[6,12] - A[9,7] * Conv_0_Relu_ts[7,12]
   - A[9,8] * Conv_0_Relu_ts[8,12] - A[9,9] * Conv_0_Relu_ts[9,12]
   + A[9,9] * Conv_1_aggregated_features[9,12] ] = 0
 Conv_1_averages_constraint[9,13]: [
   A[0,9] * Conv_1_aggregated_features[9,13]
   + A[1,9] * Conv_1_aggregated_features[9,13]
   + A[2,9] * Conv_1_aggregated_features[9,13]
   + A[3,9] * Conv_1_aggregated_features[9,13]
   + A[4,9] * Conv_1_aggregated_features[9,13]
   + A[5,9] * Conv_1_aggregated_features[9,13]
   + A[6,9] * Conv_1_aggregated_features[9,13]
   + A[7,9] * Conv_1_aggregated_features[9,13]
   + A[8,9] * Conv_1_aggregated_features[9,13]
   - A[9,0] * Conv_0_Relu_ts[0,13] - A[9,1] * Conv_0_Relu_ts[1,13]
   - A[9,2] * Conv_0_Relu_ts[2,13] - A[9,3] * Conv_0_Relu_ts[3,13]
   - A[9,4] * Conv_0_Relu_ts[4,13] - A[9,5] * Conv_0_Relu_ts[5,13]
   - A[9,6] * Conv_0_Relu_ts[6,13] - A[9,7] * Conv_0_Relu_ts[7,13]
   - A[9,8] * Conv_0_Relu_ts[8,13] - A[9,9] * Conv_0_Relu_ts[9,13]
   + A[9,9] * Conv_1_aggregated_features[9,13] ] = 0
 Conv_1_averages_constraint[9,14]: [
   A[0,9] * Conv_1_aggregated_features[9,14]
   + A[1,9] * Conv_1_aggregated_features[9,14]
   + A[2,9] * Conv_1_aggregated_features[9,14]
   + A[3,9] * Conv_1_aggregated_features[9,14]
   + A[4,9] * Conv_1_aggregated_features[9,14]
   + A[5,9] * Conv_1_aggregated_features[9,14]
   + A[6,9] * Conv_1_aggregated_features[9,14]
   + A[7,9] * Conv_1_aggregated_features[9,14]
   + A[8,9] * Conv_1_aggregated_features[9,14]
   - A[9,0] * Conv_0_Relu_ts[0,14] - A[9,1] * Conv_0_Relu_ts[1,14]
   - A[9,2] * Conv_0_Relu_ts[2,14] - A[9,3] * Conv_0_Relu_ts[3,14]
   - A[9,4] * Conv_0_Relu_ts[4,14] - A[9,5] * Conv_0_Relu_ts[5,14]
   - A[9,6] * Conv_0_Relu_ts[6,14] - A[9,7] * Conv_0_Relu_ts[7,14]
   - A[9,8] * Conv_0_Relu_ts[8,14] - A[9,9] * Conv_0_Relu_ts[9,14]
   + A[9,9] * Conv_1_aggregated_features[9,14] ] = 0
 Conv_1_averages_constraint[9,15]: [
   A[0,9] * Conv_1_aggregated_features[9,15]
   + A[1,9] * Conv_1_aggregated_features[9,15]
   + A[2,9] * Conv_1_aggregated_features[9,15]
   + A[3,9] * Conv_1_aggregated_features[9,15]
   + A[4,9] * Conv_1_aggregated_features[9,15]
   + A[5,9] * Conv_1_aggregated_features[9,15]
   + A[6,9] * Conv_1_aggregated_features[9,15]
   + A[7,9] * Conv_1_aggregated_features[9,15]
   + A[8,9] * Conv_1_aggregated_features[9,15]
   - A[9,0] * Conv_0_Relu_ts[0,15] - A[9,1] * Conv_0_Relu_ts[1,15]
   - A[9,2] * Conv_0_Relu_ts[2,15] - A[9,3] * Conv_0_Relu_ts[3,15]
   - A[9,4] * Conv_0_Relu_ts[4,15] - A[9,5] * Conv_0_Relu_ts[5,15]
   - A[9,6] * Conv_0_Relu_ts[6,15] - A[9,7] * Conv_0_Relu_ts[7,15]
   - A[9,8] * Conv_0_Relu_ts[8,15] - A[9,9] * Conv_0_Relu_ts[9,15]
   + A[9,9] * Conv_1_aggregated_features[9,15] ] = 0
 Conv_2_averages_constraint[0,0]: [ - A[0,0] * Conv_1_Relu_ts[0,0]
   + A[0,0] * Conv_2_aggregated_features[0,0]
   - A[0,1] * Conv_1_Relu_ts[1,0] - A[0,2] * Conv_1_Relu_ts[2,0]
   - A[0,3] * Conv_1_Relu_ts[3,0] - A[0,4] * Conv_1_Relu_ts[4,0]
   - A[0,5] * Conv_1_Relu_ts[5,0] - A[0,6] * Conv_1_Relu_ts[6,0]
   - A[0,7] * Conv_1_Relu_ts[7,0] - A[0,8] * Conv_1_Relu_ts[8,0]
   - A[0,9] * Conv_1_Relu_ts[9,0]
   + A[1,0] * Conv_2_aggregated_features[0,0]
   + A[2,0] * Conv_2_aggregated_features[0,0]
   + A[3,0] * Conv_2_aggregated_features[0,0]
   + A[4,0] * Conv_2_aggregated_features[0,0]
   + A[5,0] * Conv_2_aggregated_features[0,0]
   + A[6,0] * Conv_2_aggregated_features[0,0]
   + A[7,0] * Conv_2_aggregated_features[0,0]
   + A[8,0] * Conv_2_aggregated_features[0,0]
   + A[9,0] * Conv_2_aggregated_features[0,0] ] = 0
 Conv_2_averages_constraint[0,1]: [ - A[0,0] * Conv_1_Relu_ts[0,1]
   + A[0,0] * Conv_2_aggregated_features[0,1]
   - A[0,1] * Conv_1_Relu_ts[1,1] - A[0,2] * Conv_1_Relu_ts[2,1]
   - A[0,3] * Conv_1_Relu_ts[3,1] - A[0,4] * Conv_1_Relu_ts[4,1]
   - A[0,5] * Conv_1_Relu_ts[5,1] - A[0,6] * Conv_1_Relu_ts[6,1]
   - A[0,7] * Conv_1_Relu_ts[7,1] - A[0,8] * Conv_1_Relu_ts[8,1]
   - A[0,9] * Conv_1_Relu_ts[9,1]
   + A[1,0] * Conv_2_aggregated_features[0,1]
   + A[2,0] * Conv_2_aggregated_features[0,1]
   + A[3,0] * Conv_2_aggregated_features[0,1]
   + A[4,0] * Conv_2_aggregated_features[0,1]
   + A[5,0] * Conv_2_aggregated_features[0,1]
   + A[6,0] * Conv_2_aggregated_features[0,1]
   + A[7,0] * Conv_2_aggregated_features[0,1]
   + A[8,0] * Conv_2_aggregated_features[0,1]
   + A[9,0] * Conv_2_aggregated_features[0,1] ] = 0
 Conv_2_averages_constraint[0,2]: [ - A[0,0] * Conv_1_Relu_ts[0,2]
   + A[0,0] * Conv_2_aggregated_features[0,2]
   - A[0,1] * Conv_1_Relu_ts[1,2] - A[0,2] * Conv_1_Relu_ts[2,2]
   - A[0,3] * Conv_1_Relu_ts[3,2] - A[0,4] * Conv_1_Relu_ts[4,2]
   - A[0,5] * Conv_1_Relu_ts[5,2] - A[0,6] * Conv_1_Relu_ts[6,2]
   - A[0,7] * Conv_1_Relu_ts[7,2] - A[0,8] * Conv_1_Relu_ts[8,2]
   - A[0,9] * Conv_1_Relu_ts[9,2]
   + A[1,0] * Conv_2_aggregated_features[0,2]
   + A[2,0] * Conv_2_aggregated_features[0,2]
   + A[3,0] * Conv_2_aggregated_features[0,2]
   + A[4,0] * Conv_2_aggregated_features[0,2]
   + A[5,0] * Conv_2_aggregated_features[0,2]
   + A[6,0] * Conv_2_aggregated_features[0,2]
   + A[7,0] * Conv_2_aggregated_features[0,2]
   + A[8,0] * Conv_2_aggregated_features[0,2]
   + A[9,0] * Conv_2_aggregated_features[0,2] ] = 0
 Conv_2_averages_constraint[0,3]: [ - A[0,0] * Conv_1_Relu_ts[0,3]
   + A[0,0] * Conv_2_aggregated_features[0,3]
   - A[0,1] * Conv_1_Relu_ts[1,3] - A[0,2] * Conv_1_Relu_ts[2,3]
   - A[0,3] * Conv_1_Relu_ts[3,3] - A[0,4] * Conv_1_Relu_ts[4,3]
   - A[0,5] * Conv_1_Relu_ts[5,3] - A[0,6] * Conv_1_Relu_ts[6,3]
   - A[0,7] * Conv_1_Relu_ts[7,3] - A[0,8] * Conv_1_Relu_ts[8,3]
   - A[0,9] * Conv_1_Relu_ts[9,3]
   + A[1,0] * Conv_2_aggregated_features[0,3]
   + A[2,0] * Conv_2_aggregated_features[0,3]
   + A[3,0] * Conv_2_aggregated_features[0,3]
   + A[4,0] * Conv_2_aggregated_features[0,3]
   + A[5,0] * Conv_2_aggregated_features[0,3]
   + A[6,0] * Conv_2_aggregated_features[0,3]
   + A[7,0] * Conv_2_aggregated_features[0,3]
   + A[8,0] * Conv_2_aggregated_features[0,3]
   + A[9,0] * Conv_2_aggregated_features[0,3] ] = 0
 Conv_2_averages_constraint[0,4]: [ - A[0,0] * Conv_1_Relu_ts[0,4]
   + A[0,0] * Conv_2_aggregated_features[0,4]
   - A[0,1] * Conv_1_Relu_ts[1,4] - A[0,2] * Conv_1_Relu_ts[2,4]
   - A[0,3] * Conv_1_Relu_ts[3,4] - A[0,4] * Conv_1_Relu_ts[4,4]
   - A[0,5] * Conv_1_Relu_ts[5,4] - A[0,6] * Conv_1_Relu_ts[6,4]
   - A[0,7] * Conv_1_Relu_ts[7,4] - A[0,8] * Conv_1_Relu_ts[8,4]
   - A[0,9] * Conv_1_Relu_ts[9,4]
   + A[1,0] * Conv_2_aggregated_features[0,4]
   + A[2,0] * Conv_2_aggregated_features[0,4]
   + A[3,0] * Conv_2_aggregated_features[0,4]
   + A[4,0] * Conv_2_aggregated_features[0,4]
   + A[5,0] * Conv_2_aggregated_features[0,4]
   + A[6,0] * Conv_2_aggregated_features[0,4]
   + A[7,0] * Conv_2_aggregated_features[0,4]
   + A[8,0] * Conv_2_aggregated_features[0,4]
   + A[9,0] * Conv_2_aggregated_features[0,4] ] = 0
 Conv_2_averages_constraint[0,5]: [ - A[0,0] * Conv_1_Relu_ts[0,5]
   + A[0,0] * Conv_2_aggregated_features[0,5]
   - A[0,1] * Conv_1_Relu_ts[1,5] - A[0,2] * Conv_1_Relu_ts[2,5]
   - A[0,3] * Conv_1_Relu_ts[3,5] - A[0,4] * Conv_1_Relu_ts[4,5]
   - A[0,5] * Conv_1_Relu_ts[5,5] - A[0,6] * Conv_1_Relu_ts[6,5]
   - A[0,7] * Conv_1_Relu_ts[7,5] - A[0,8] * Conv_1_Relu_ts[8,5]
   - A[0,9] * Conv_1_Relu_ts[9,5]
   + A[1,0] * Conv_2_aggregated_features[0,5]
   + A[2,0] * Conv_2_aggregated_features[0,5]
   + A[3,0] * Conv_2_aggregated_features[0,5]
   + A[4,0] * Conv_2_aggregated_features[0,5]
   + A[5,0] * Conv_2_aggregated_features[0,5]
   + A[6,0] * Conv_2_aggregated_features[0,5]
   + A[7,0] * Conv_2_aggregated_features[0,5]
   + A[8,0] * Conv_2_aggregated_features[0,5]
   + A[9,0] * Conv_2_aggregated_features[0,5] ] = 0
 Conv_2_averages_constraint[0,6]: [ - A[0,0] * Conv_1_Relu_ts[0,6]
   + A[0,0] * Conv_2_aggregated_features[0,6]
   - A[0,1] * Conv_1_Relu_ts[1,6] - A[0,2] * Conv_1_Relu_ts[2,6]
   - A[0,3] * Conv_1_Relu_ts[3,6] - A[0,4] * Conv_1_Relu_ts[4,6]
   - A[0,5] * Conv_1_Relu_ts[5,6] - A[0,6] * Conv_1_Relu_ts[6,6]
   - A[0,7] * Conv_1_Relu_ts[7,6] - A[0,8] * Conv_1_Relu_ts[8,6]
   - A[0,9] * Conv_1_Relu_ts[9,6]
   + A[1,0] * Conv_2_aggregated_features[0,6]
   + A[2,0] * Conv_2_aggregated_features[0,6]
   + A[3,0] * Conv_2_aggregated_features[0,6]
   + A[4,0] * Conv_2_aggregated_features[0,6]
   + A[5,0] * Conv_2_aggregated_features[0,6]
   + A[6,0] * Conv_2_aggregated_features[0,6]
   + A[7,0] * Conv_2_aggregated_features[0,6]
   + A[8,0] * Conv_2_aggregated_features[0,6]
   + A[9,0] * Conv_2_aggregated_features[0,6] ] = 0
 Conv_2_averages_constraint[0,7]: [ - A[0,0] * Conv_1_Relu_ts[0,7]
   + A[0,0] * Conv_2_aggregated_features[0,7]
   - A[0,1] * Conv_1_Relu_ts[1,7] - A[0,2] * Conv_1_Relu_ts[2,7]
   - A[0,3] * Conv_1_Relu_ts[3,7] - A[0,4] * Conv_1_Relu_ts[4,7]
   - A[0,5] * Conv_1_Relu_ts[5,7] - A[0,6] * Conv_1_Relu_ts[6,7]
   - A[0,7] * Conv_1_Relu_ts[7,7] - A[0,8] * Conv_1_Relu_ts[8,7]
   - A[0,9] * Conv_1_Relu_ts[9,7]
   + A[1,0] * Conv_2_aggregated_features[0,7]
   + A[2,0] * Conv_2_aggregated_features[0,7]
   + A[3,0] * Conv_2_aggregated_features[0,7]
   + A[4,0] * Conv_2_aggregated_features[0,7]
   + A[5,0] * Conv_2_aggregated_features[0,7]
   + A[6,0] * Conv_2_aggregated_features[0,7]
   + A[7,0] * Conv_2_aggregated_features[0,7]
   + A[8,0] * Conv_2_aggregated_features[0,7]
   + A[9,0] * Conv_2_aggregated_features[0,7] ] = 0
 Conv_2_averages_constraint[0,8]: [ - A[0,0] * Conv_1_Relu_ts[0,8]
   + A[0,0] * Conv_2_aggregated_features[0,8]
   - A[0,1] * Conv_1_Relu_ts[1,8] - A[0,2] * Conv_1_Relu_ts[2,8]
   - A[0,3] * Conv_1_Relu_ts[3,8] - A[0,4] * Conv_1_Relu_ts[4,8]
   - A[0,5] * Conv_1_Relu_ts[5,8] - A[0,6] * Conv_1_Relu_ts[6,8]
   - A[0,7] * Conv_1_Relu_ts[7,8] - A[0,8] * Conv_1_Relu_ts[8,8]
   - A[0,9] * Conv_1_Relu_ts[9,8]
   + A[1,0] * Conv_2_aggregated_features[0,8]
   + A[2,0] * Conv_2_aggregated_features[0,8]
   + A[3,0] * Conv_2_aggregated_features[0,8]
   + A[4,0] * Conv_2_aggregated_features[0,8]
   + A[5,0] * Conv_2_aggregated_features[0,8]
   + A[6,0] * Conv_2_aggregated_features[0,8]
   + A[7,0] * Conv_2_aggregated_features[0,8]
   + A[8,0] * Conv_2_aggregated_features[0,8]
   + A[9,0] * Conv_2_aggregated_features[0,8] ] = 0
 Conv_2_averages_constraint[0,9]: [ - A[0,0] * Conv_1_Relu_ts[0,9]
   + A[0,0] * Conv_2_aggregated_features[0,9]
   - A[0,1] * Conv_1_Relu_ts[1,9] - A[0,2] * Conv_1_Relu_ts[2,9]
   - A[0,3] * Conv_1_Relu_ts[3,9] - A[0,4] * Conv_1_Relu_ts[4,9]
   - A[0,5] * Conv_1_Relu_ts[5,9] - A[0,6] * Conv_1_Relu_ts[6,9]
   - A[0,7] * Conv_1_Relu_ts[7,9] - A[0,8] * Conv_1_Relu_ts[8,9]
   - A[0,9] * Conv_1_Relu_ts[9,9]
   + A[1,0] * Conv_2_aggregated_features[0,9]
   + A[2,0] * Conv_2_aggregated_features[0,9]
   + A[3,0] * Conv_2_aggregated_features[0,9]
   + A[4,0] * Conv_2_aggregated_features[0,9]
   + A[5,0] * Conv_2_aggregated_features[0,9]
   + A[6,0] * Conv_2_aggregated_features[0,9]
   + A[7,0] * Conv_2_aggregated_features[0,9]
   + A[8,0] * Conv_2_aggregated_features[0,9]
   + A[9,0] * Conv_2_aggregated_features[0,9] ] = 0
 Conv_2_averages_constraint[0,10]: [ - A[0,0] * Conv_1_Relu_ts[0,10]
   + A[0,0] * Conv_2_aggregated_features[0,10]
   - A[0,1] * Conv_1_Relu_ts[1,10] - A[0,2] * Conv_1_Relu_ts[2,10]
   - A[0,3] * Conv_1_Relu_ts[3,10] - A[0,4] * Conv_1_Relu_ts[4,10]
   - A[0,5] * Conv_1_Relu_ts[5,10] - A[0,6] * Conv_1_Relu_ts[6,10]
   - A[0,7] * Conv_1_Relu_ts[7,10] - A[0,8] * Conv_1_Relu_ts[8,10]
   - A[0,9] * Conv_1_Relu_ts[9,10]
   + A[1,0] * Conv_2_aggregated_features[0,10]
   + A[2,0] * Conv_2_aggregated_features[0,10]
   + A[3,0] * Conv_2_aggregated_features[0,10]
   + A[4,0] * Conv_2_aggregated_features[0,10]
   + A[5,0] * Conv_2_aggregated_features[0,10]
   + A[6,0] * Conv_2_aggregated_features[0,10]
   + A[7,0] * Conv_2_aggregated_features[0,10]
   + A[8,0] * Conv_2_aggregated_features[0,10]
   + A[9,0] * Conv_2_aggregated_features[0,10] ] = 0
 Conv_2_averages_constraint[0,11]: [ - A[0,0] * Conv_1_Relu_ts[0,11]
   + A[0,0] * Conv_2_aggregated_features[0,11]
   - A[0,1] * Conv_1_Relu_ts[1,11] - A[0,2] * Conv_1_Relu_ts[2,11]
   - A[0,3] * Conv_1_Relu_ts[3,11] - A[0,4] * Conv_1_Relu_ts[4,11]
   - A[0,5] * Conv_1_Relu_ts[5,11] - A[0,6] * Conv_1_Relu_ts[6,11]
   - A[0,7] * Conv_1_Relu_ts[7,11] - A[0,8] * Conv_1_Relu_ts[8,11]
   - A[0,9] * Conv_1_Relu_ts[9,11]
   + A[1,0] * Conv_2_aggregated_features[0,11]
   + A[2,0] * Conv_2_aggregated_features[0,11]
   + A[3,0] * Conv_2_aggregated_features[0,11]
   + A[4,0] * Conv_2_aggregated_features[0,11]
   + A[5,0] * Conv_2_aggregated_features[0,11]
   + A[6,0] * Conv_2_aggregated_features[0,11]
   + A[7,0] * Conv_2_aggregated_features[0,11]
   + A[8,0] * Conv_2_aggregated_features[0,11]
   + A[9,0] * Conv_2_aggregated_features[0,11] ] = 0
 Conv_2_averages_constraint[0,12]: [ - A[0,0] * Conv_1_Relu_ts[0,12]
   + A[0,0] * Conv_2_aggregated_features[0,12]
   - A[0,1] * Conv_1_Relu_ts[1,12] - A[0,2] * Conv_1_Relu_ts[2,12]
   - A[0,3] * Conv_1_Relu_ts[3,12] - A[0,4] * Conv_1_Relu_ts[4,12]
   - A[0,5] * Conv_1_Relu_ts[5,12] - A[0,6] * Conv_1_Relu_ts[6,12]
   - A[0,7] * Conv_1_Relu_ts[7,12] - A[0,8] * Conv_1_Relu_ts[8,12]
   - A[0,9] * Conv_1_Relu_ts[9,12]
   + A[1,0] * Conv_2_aggregated_features[0,12]
   + A[2,0] * Conv_2_aggregated_features[0,12]
   + A[3,0] * Conv_2_aggregated_features[0,12]
   + A[4,0] * Conv_2_aggregated_features[0,12]
   + A[5,0] * Conv_2_aggregated_features[0,12]
   + A[6,0] * Conv_2_aggregated_features[0,12]
   + A[7,0] * Conv_2_aggregated_features[0,12]
   + A[8,0] * Conv_2_aggregated_features[0,12]
   + A[9,0] * Conv_2_aggregated_features[0,12] ] = 0
 Conv_2_averages_constraint[0,13]: [ - A[0,0] * Conv_1_Relu_ts[0,13]
   + A[0,0] * Conv_2_aggregated_features[0,13]
   - A[0,1] * Conv_1_Relu_ts[1,13] - A[0,2] * Conv_1_Relu_ts[2,13]
   - A[0,3] * Conv_1_Relu_ts[3,13] - A[0,4] * Conv_1_Relu_ts[4,13]
   - A[0,5] * Conv_1_Relu_ts[5,13] - A[0,6] * Conv_1_Relu_ts[6,13]
   - A[0,7] * Conv_1_Relu_ts[7,13] - A[0,8] * Conv_1_Relu_ts[8,13]
   - A[0,9] * Conv_1_Relu_ts[9,13]
   + A[1,0] * Conv_2_aggregated_features[0,13]
   + A[2,0] * Conv_2_aggregated_features[0,13]
   + A[3,0] * Conv_2_aggregated_features[0,13]
   + A[4,0] * Conv_2_aggregated_features[0,13]
   + A[5,0] * Conv_2_aggregated_features[0,13]
   + A[6,0] * Conv_2_aggregated_features[0,13]
   + A[7,0] * Conv_2_aggregated_features[0,13]
   + A[8,0] * Conv_2_aggregated_features[0,13]
   + A[9,0] * Conv_2_aggregated_features[0,13] ] = 0
 Conv_2_averages_constraint[0,14]: [ - A[0,0] * Conv_1_Relu_ts[0,14]
   + A[0,0] * Conv_2_aggregated_features[0,14]
   - A[0,1] * Conv_1_Relu_ts[1,14] - A[0,2] * Conv_1_Relu_ts[2,14]
   - A[0,3] * Conv_1_Relu_ts[3,14] - A[0,4] * Conv_1_Relu_ts[4,14]
   - A[0,5] * Conv_1_Relu_ts[5,14] - A[0,6] * Conv_1_Relu_ts[6,14]
   - A[0,7] * Conv_1_Relu_ts[7,14] - A[0,8] * Conv_1_Relu_ts[8,14]
   - A[0,9] * Conv_1_Relu_ts[9,14]
   + A[1,0] * Conv_2_aggregated_features[0,14]
   + A[2,0] * Conv_2_aggregated_features[0,14]
   + A[3,0] * Conv_2_aggregated_features[0,14]
   + A[4,0] * Conv_2_aggregated_features[0,14]
   + A[5,0] * Conv_2_aggregated_features[0,14]
   + A[6,0] * Conv_2_aggregated_features[0,14]
   + A[7,0] * Conv_2_aggregated_features[0,14]
   + A[8,0] * Conv_2_aggregated_features[0,14]
   + A[9,0] * Conv_2_aggregated_features[0,14] ] = 0
 Conv_2_averages_constraint[0,15]: [ - A[0,0] * Conv_1_Relu_ts[0,15]
   + A[0,0] * Conv_2_aggregated_features[0,15]
   - A[0,1] * Conv_1_Relu_ts[1,15] - A[0,2] * Conv_1_Relu_ts[2,15]
   - A[0,3] * Conv_1_Relu_ts[3,15] - A[0,4] * Conv_1_Relu_ts[4,15]
   - A[0,5] * Conv_1_Relu_ts[5,15] - A[0,6] * Conv_1_Relu_ts[6,15]
   - A[0,7] * Conv_1_Relu_ts[7,15] - A[0,8] * Conv_1_Relu_ts[8,15]
   - A[0,9] * Conv_1_Relu_ts[9,15]
   + A[1,0] * Conv_2_aggregated_features[0,15]
   + A[2,0] * Conv_2_aggregated_features[0,15]
   + A[3,0] * Conv_2_aggregated_features[0,15]
   + A[4,0] * Conv_2_aggregated_features[0,15]
   + A[5,0] * Conv_2_aggregated_features[0,15]
   + A[6,0] * Conv_2_aggregated_features[0,15]
   + A[7,0] * Conv_2_aggregated_features[0,15]
   + A[8,0] * Conv_2_aggregated_features[0,15]
   + A[9,0] * Conv_2_aggregated_features[0,15] ] = 0
 Conv_2_averages_constraint[1,0]: [
   A[0,1] * Conv_2_aggregated_features[1,0] - A[1,0] * Conv_1_Relu_ts[0,0]
   - A[1,1] * Conv_1_Relu_ts[1,0]
   + A[1,1] * Conv_2_aggregated_features[1,0]
   - A[1,2] * Conv_1_Relu_ts[2,0] - A[1,3] * Conv_1_Relu_ts[3,0]
   - A[1,4] * Conv_1_Relu_ts[4,0] - A[1,5] * Conv_1_Relu_ts[5,0]
   - A[1,6] * Conv_1_Relu_ts[6,0] - A[1,7] * Conv_1_Relu_ts[7,0]
   - A[1,8] * Conv_1_Relu_ts[8,0] - A[1,9] * Conv_1_Relu_ts[9,0]
   + A[2,1] * Conv_2_aggregated_features[1,0]
   + A[3,1] * Conv_2_aggregated_features[1,0]
   + A[4,1] * Conv_2_aggregated_features[1,0]
   + A[5,1] * Conv_2_aggregated_features[1,0]
   + A[6,1] * Conv_2_aggregated_features[1,0]
   + A[7,1] * Conv_2_aggregated_features[1,0]
   + A[8,1] * Conv_2_aggregated_features[1,0]
   + A[9,1] * Conv_2_aggregated_features[1,0] ] = 0
 Conv_2_averages_constraint[1,1]: [
   A[0,1] * Conv_2_aggregated_features[1,1] - A[1,0] * Conv_1_Relu_ts[0,1]
   - A[1,1] * Conv_1_Relu_ts[1,1]
   + A[1,1] * Conv_2_aggregated_features[1,1]
   - A[1,2] * Conv_1_Relu_ts[2,1] - A[1,3] * Conv_1_Relu_ts[3,1]
   - A[1,4] * Conv_1_Relu_ts[4,1] - A[1,5] * Conv_1_Relu_ts[5,1]
   - A[1,6] * Conv_1_Relu_ts[6,1] - A[1,7] * Conv_1_Relu_ts[7,1]
   - A[1,8] * Conv_1_Relu_ts[8,1] - A[1,9] * Conv_1_Relu_ts[9,1]
   + A[2,1] * Conv_2_aggregated_features[1,1]
   + A[3,1] * Conv_2_aggregated_features[1,1]
   + A[4,1] * Conv_2_aggregated_features[1,1]
   + A[5,1] * Conv_2_aggregated_features[1,1]
   + A[6,1] * Conv_2_aggregated_features[1,1]
   + A[7,1] * Conv_2_aggregated_features[1,1]
   + A[8,1] * Conv_2_aggregated_features[1,1]
   + A[9,1] * Conv_2_aggregated_features[1,1] ] = 0
 Conv_2_averages_constraint[1,2]: [
   A[0,1] * Conv_2_aggregated_features[1,2] - A[1,0] * Conv_1_Relu_ts[0,2]
   - A[1,1] * Conv_1_Relu_ts[1,2]
   + A[1,1] * Conv_2_aggregated_features[1,2]
   - A[1,2] * Conv_1_Relu_ts[2,2] - A[1,3] * Conv_1_Relu_ts[3,2]
   - A[1,4] * Conv_1_Relu_ts[4,2] - A[1,5] * Conv_1_Relu_ts[5,2]
   - A[1,6] * Conv_1_Relu_ts[6,2] - A[1,7] * Conv_1_Relu_ts[7,2]
   - A[1,8] * Conv_1_Relu_ts[8,2] - A[1,9] * Conv_1_Relu_ts[9,2]
   + A[2,1] * Conv_2_aggregated_features[1,2]
   + A[3,1] * Conv_2_aggregated_features[1,2]
   + A[4,1] * Conv_2_aggregated_features[1,2]
   + A[5,1] * Conv_2_aggregated_features[1,2]
   + A[6,1] * Conv_2_aggregated_features[1,2]
   + A[7,1] * Conv_2_aggregated_features[1,2]
   + A[8,1] * Conv_2_aggregated_features[1,2]
   + A[9,1] * Conv_2_aggregated_features[1,2] ] = 0
 Conv_2_averages_constraint[1,3]: [
   A[0,1] * Conv_2_aggregated_features[1,3] - A[1,0] * Conv_1_Relu_ts[0,3]
   - A[1,1] * Conv_1_Relu_ts[1,3]
   + A[1,1] * Conv_2_aggregated_features[1,3]
   - A[1,2] * Conv_1_Relu_ts[2,3] - A[1,3] * Conv_1_Relu_ts[3,3]
   - A[1,4] * Conv_1_Relu_ts[4,3] - A[1,5] * Conv_1_Relu_ts[5,3]
   - A[1,6] * Conv_1_Relu_ts[6,3] - A[1,7] * Conv_1_Relu_ts[7,3]
   - A[1,8] * Conv_1_Relu_ts[8,3] - A[1,9] * Conv_1_Relu_ts[9,3]
   + A[2,1] * Conv_2_aggregated_features[1,3]
   + A[3,1] * Conv_2_aggregated_features[1,3]
   + A[4,1] * Conv_2_aggregated_features[1,3]
   + A[5,1] * Conv_2_aggregated_features[1,3]
   + A[6,1] * Conv_2_aggregated_features[1,3]
   + A[7,1] * Conv_2_aggregated_features[1,3]
   + A[8,1] * Conv_2_aggregated_features[1,3]
   + A[9,1] * Conv_2_aggregated_features[1,3] ] = 0
 Conv_2_averages_constraint[1,4]: [
   A[0,1] * Conv_2_aggregated_features[1,4] - A[1,0] * Conv_1_Relu_ts[0,4]
   - A[1,1] * Conv_1_Relu_ts[1,4]
   + A[1,1] * Conv_2_aggregated_features[1,4]
   - A[1,2] * Conv_1_Relu_ts[2,4] - A[1,3] * Conv_1_Relu_ts[3,4]
   - A[1,4] * Conv_1_Relu_ts[4,4] - A[1,5] * Conv_1_Relu_ts[5,4]
   - A[1,6] * Conv_1_Relu_ts[6,4] - A[1,7] * Conv_1_Relu_ts[7,4]
   - A[1,8] * Conv_1_Relu_ts[8,4] - A[1,9] * Conv_1_Relu_ts[9,4]
   + A[2,1] * Conv_2_aggregated_features[1,4]
   + A[3,1] * Conv_2_aggregated_features[1,4]
   + A[4,1] * Conv_2_aggregated_features[1,4]
   + A[5,1] * Conv_2_aggregated_features[1,4]
   + A[6,1] * Conv_2_aggregated_features[1,4]
   + A[7,1] * Conv_2_aggregated_features[1,4]
   + A[8,1] * Conv_2_aggregated_features[1,4]
   + A[9,1] * Conv_2_aggregated_features[1,4] ] = 0
 Conv_2_averages_constraint[1,5]: [
   A[0,1] * Conv_2_aggregated_features[1,5] - A[1,0] * Conv_1_Relu_ts[0,5]
   - A[1,1] * Conv_1_Relu_ts[1,5]
   + A[1,1] * Conv_2_aggregated_features[1,5]
   - A[1,2] * Conv_1_Relu_ts[2,5] - A[1,3] * Conv_1_Relu_ts[3,5]
   - A[1,4] * Conv_1_Relu_ts[4,5] - A[1,5] * Conv_1_Relu_ts[5,5]
   - A[1,6] * Conv_1_Relu_ts[6,5] - A[1,7] * Conv_1_Relu_ts[7,5]
   - A[1,8] * Conv_1_Relu_ts[8,5] - A[1,9] * Conv_1_Relu_ts[9,5]
   + A[2,1] * Conv_2_aggregated_features[1,5]
   + A[3,1] * Conv_2_aggregated_features[1,5]
   + A[4,1] * Conv_2_aggregated_features[1,5]
   + A[5,1] * Conv_2_aggregated_features[1,5]
   + A[6,1] * Conv_2_aggregated_features[1,5]
   + A[7,1] * Conv_2_aggregated_features[1,5]
   + A[8,1] * Conv_2_aggregated_features[1,5]
   + A[9,1] * Conv_2_aggregated_features[1,5] ] = 0
 Conv_2_averages_constraint[1,6]: [
   A[0,1] * Conv_2_aggregated_features[1,6] - A[1,0] * Conv_1_Relu_ts[0,6]
   - A[1,1] * Conv_1_Relu_ts[1,6]
   + A[1,1] * Conv_2_aggregated_features[1,6]
   - A[1,2] * Conv_1_Relu_ts[2,6] - A[1,3] * Conv_1_Relu_ts[3,6]
   - A[1,4] * Conv_1_Relu_ts[4,6] - A[1,5] * Conv_1_Relu_ts[5,6]
   - A[1,6] * Conv_1_Relu_ts[6,6] - A[1,7] * Conv_1_Relu_ts[7,6]
   - A[1,8] * Conv_1_Relu_ts[8,6] - A[1,9] * Conv_1_Relu_ts[9,6]
   + A[2,1] * Conv_2_aggregated_features[1,6]
   + A[3,1] * Conv_2_aggregated_features[1,6]
   + A[4,1] * Conv_2_aggregated_features[1,6]
   + A[5,1] * Conv_2_aggregated_features[1,6]
   + A[6,1] * Conv_2_aggregated_features[1,6]
   + A[7,1] * Conv_2_aggregated_features[1,6]
   + A[8,1] * Conv_2_aggregated_features[1,6]
   + A[9,1] * Conv_2_aggregated_features[1,6] ] = 0
 Conv_2_averages_constraint[1,7]: [
   A[0,1] * Conv_2_aggregated_features[1,7] - A[1,0] * Conv_1_Relu_ts[0,7]
   - A[1,1] * Conv_1_Relu_ts[1,7]
   + A[1,1] * Conv_2_aggregated_features[1,7]
   - A[1,2] * Conv_1_Relu_ts[2,7] - A[1,3] * Conv_1_Relu_ts[3,7]
   - A[1,4] * Conv_1_Relu_ts[4,7] - A[1,5] * Conv_1_Relu_ts[5,7]
   - A[1,6] * Conv_1_Relu_ts[6,7] - A[1,7] * Conv_1_Relu_ts[7,7]
   - A[1,8] * Conv_1_Relu_ts[8,7] - A[1,9] * Conv_1_Relu_ts[9,7]
   + A[2,1] * Conv_2_aggregated_features[1,7]
   + A[3,1] * Conv_2_aggregated_features[1,7]
   + A[4,1] * Conv_2_aggregated_features[1,7]
   + A[5,1] * Conv_2_aggregated_features[1,7]
   + A[6,1] * Conv_2_aggregated_features[1,7]
   + A[7,1] * Conv_2_aggregated_features[1,7]
   + A[8,1] * Conv_2_aggregated_features[1,7]
   + A[9,1] * Conv_2_aggregated_features[1,7] ] = 0
 Conv_2_averages_constraint[1,8]: [
   A[0,1] * Conv_2_aggregated_features[1,8] - A[1,0] * Conv_1_Relu_ts[0,8]
   - A[1,1] * Conv_1_Relu_ts[1,8]
   + A[1,1] * Conv_2_aggregated_features[1,8]
   - A[1,2] * Conv_1_Relu_ts[2,8] - A[1,3] * Conv_1_Relu_ts[3,8]
   - A[1,4] * Conv_1_Relu_ts[4,8] - A[1,5] * Conv_1_Relu_ts[5,8]
   - A[1,6] * Conv_1_Relu_ts[6,8] - A[1,7] * Conv_1_Relu_ts[7,8]
   - A[1,8] * Conv_1_Relu_ts[8,8] - A[1,9] * Conv_1_Relu_ts[9,8]
   + A[2,1] * Conv_2_aggregated_features[1,8]
   + A[3,1] * Conv_2_aggregated_features[1,8]
   + A[4,1] * Conv_2_aggregated_features[1,8]
   + A[5,1] * Conv_2_aggregated_features[1,8]
   + A[6,1] * Conv_2_aggregated_features[1,8]
   + A[7,1] * Conv_2_aggregated_features[1,8]
   + A[8,1] * Conv_2_aggregated_features[1,8]
   + A[9,1] * Conv_2_aggregated_features[1,8] ] = 0
 Conv_2_averages_constraint[1,9]: [
   A[0,1] * Conv_2_aggregated_features[1,9] - A[1,0] * Conv_1_Relu_ts[0,9]
   - A[1,1] * Conv_1_Relu_ts[1,9]
   + A[1,1] * Conv_2_aggregated_features[1,9]
   - A[1,2] * Conv_1_Relu_ts[2,9] - A[1,3] * Conv_1_Relu_ts[3,9]
   - A[1,4] * Conv_1_Relu_ts[4,9] - A[1,5] * Conv_1_Relu_ts[5,9]
   - A[1,6] * Conv_1_Relu_ts[6,9] - A[1,7] * Conv_1_Relu_ts[7,9]
   - A[1,8] * Conv_1_Relu_ts[8,9] - A[1,9] * Conv_1_Relu_ts[9,9]
   + A[2,1] * Conv_2_aggregated_features[1,9]
   + A[3,1] * Conv_2_aggregated_features[1,9]
   + A[4,1] * Conv_2_aggregated_features[1,9]
   + A[5,1] * Conv_2_aggregated_features[1,9]
   + A[6,1] * Conv_2_aggregated_features[1,9]
   + A[7,1] * Conv_2_aggregated_features[1,9]
   + A[8,1] * Conv_2_aggregated_features[1,9]
   + A[9,1] * Conv_2_aggregated_features[1,9] ] = 0
 Conv_2_averages_constraint[1,10]: [
   A[0,1] * Conv_2_aggregated_features[1,10]
   - A[1,0] * Conv_1_Relu_ts[0,10] - A[1,1] * Conv_1_Relu_ts[1,10]
   + A[1,1] * Conv_2_aggregated_features[1,10]
   - A[1,2] * Conv_1_Relu_ts[2,10] - A[1,3] * Conv_1_Relu_ts[3,10]
   - A[1,4] * Conv_1_Relu_ts[4,10] - A[1,5] * Conv_1_Relu_ts[5,10]
   - A[1,6] * Conv_1_Relu_ts[6,10] - A[1,7] * Conv_1_Relu_ts[7,10]
   - A[1,8] * Conv_1_Relu_ts[8,10] - A[1,9] * Conv_1_Relu_ts[9,10]
   + A[2,1] * Conv_2_aggregated_features[1,10]
   + A[3,1] * Conv_2_aggregated_features[1,10]
   + A[4,1] * Conv_2_aggregated_features[1,10]
   + A[5,1] * Conv_2_aggregated_features[1,10]
   + A[6,1] * Conv_2_aggregated_features[1,10]
   + A[7,1] * Conv_2_aggregated_features[1,10]
   + A[8,1] * Conv_2_aggregated_features[1,10]
   + A[9,1] * Conv_2_aggregated_features[1,10] ] = 0
 Conv_2_averages_constraint[1,11]: [
   A[0,1] * Conv_2_aggregated_features[1,11]
   - A[1,0] * Conv_1_Relu_ts[0,11] - A[1,1] * Conv_1_Relu_ts[1,11]
   + A[1,1] * Conv_2_aggregated_features[1,11]
   - A[1,2] * Conv_1_Relu_ts[2,11] - A[1,3] * Conv_1_Relu_ts[3,11]
   - A[1,4] * Conv_1_Relu_ts[4,11] - A[1,5] * Conv_1_Relu_ts[5,11]
   - A[1,6] * Conv_1_Relu_ts[6,11] - A[1,7] * Conv_1_Relu_ts[7,11]
   - A[1,8] * Conv_1_Relu_ts[8,11] - A[1,9] * Conv_1_Relu_ts[9,11]
   + A[2,1] * Conv_2_aggregated_features[1,11]
   + A[3,1] * Conv_2_aggregated_features[1,11]
   + A[4,1] * Conv_2_aggregated_features[1,11]
   + A[5,1] * Conv_2_aggregated_features[1,11]
   + A[6,1] * Conv_2_aggregated_features[1,11]
   + A[7,1] * Conv_2_aggregated_features[1,11]
   + A[8,1] * Conv_2_aggregated_features[1,11]
   + A[9,1] * Conv_2_aggregated_features[1,11] ] = 0
 Conv_2_averages_constraint[1,12]: [
   A[0,1] * Conv_2_aggregated_features[1,12]
   - A[1,0] * Conv_1_Relu_ts[0,12] - A[1,1] * Conv_1_Relu_ts[1,12]
   + A[1,1] * Conv_2_aggregated_features[1,12]
   - A[1,2] * Conv_1_Relu_ts[2,12] - A[1,3] * Conv_1_Relu_ts[3,12]
   - A[1,4] * Conv_1_Relu_ts[4,12] - A[1,5] * Conv_1_Relu_ts[5,12]
   - A[1,6] * Conv_1_Relu_ts[6,12] - A[1,7] * Conv_1_Relu_ts[7,12]
   - A[1,8] * Conv_1_Relu_ts[8,12] - A[1,9] * Conv_1_Relu_ts[9,12]
   + A[2,1] * Conv_2_aggregated_features[1,12]
   + A[3,1] * Conv_2_aggregated_features[1,12]
   + A[4,1] * Conv_2_aggregated_features[1,12]
   + A[5,1] * Conv_2_aggregated_features[1,12]
   + A[6,1] * Conv_2_aggregated_features[1,12]
   + A[7,1] * Conv_2_aggregated_features[1,12]
   + A[8,1] * Conv_2_aggregated_features[1,12]
   + A[9,1] * Conv_2_aggregated_features[1,12] ] = 0
 Conv_2_averages_constraint[1,13]: [
   A[0,1] * Conv_2_aggregated_features[1,13]
   - A[1,0] * Conv_1_Relu_ts[0,13] - A[1,1] * Conv_1_Relu_ts[1,13]
   + A[1,1] * Conv_2_aggregated_features[1,13]
   - A[1,2] * Conv_1_Relu_ts[2,13] - A[1,3] * Conv_1_Relu_ts[3,13]
   - A[1,4] * Conv_1_Relu_ts[4,13] - A[1,5] * Conv_1_Relu_ts[5,13]
   - A[1,6] * Conv_1_Relu_ts[6,13] - A[1,7] * Conv_1_Relu_ts[7,13]
   - A[1,8] * Conv_1_Relu_ts[8,13] - A[1,9] * Conv_1_Relu_ts[9,13]
   + A[2,1] * Conv_2_aggregated_features[1,13]
   + A[3,1] * Conv_2_aggregated_features[1,13]
   + A[4,1] * Conv_2_aggregated_features[1,13]
   + A[5,1] * Conv_2_aggregated_features[1,13]
   + A[6,1] * Conv_2_aggregated_features[1,13]
   + A[7,1] * Conv_2_aggregated_features[1,13]
   + A[8,1] * Conv_2_aggregated_features[1,13]
   + A[9,1] * Conv_2_aggregated_features[1,13] ] = 0
 Conv_2_averages_constraint[1,14]: [
   A[0,1] * Conv_2_aggregated_features[1,14]
   - A[1,0] * Conv_1_Relu_ts[0,14] - A[1,1] * Conv_1_Relu_ts[1,14]
   + A[1,1] * Conv_2_aggregated_features[1,14]
   - A[1,2] * Conv_1_Relu_ts[2,14] - A[1,3] * Conv_1_Relu_ts[3,14]
   - A[1,4] * Conv_1_Relu_ts[4,14] - A[1,5] * Conv_1_Relu_ts[5,14]
   - A[1,6] * Conv_1_Relu_ts[6,14] - A[1,7] * Conv_1_Relu_ts[7,14]
   - A[1,8] * Conv_1_Relu_ts[8,14] - A[1,9] * Conv_1_Relu_ts[9,14]
   + A[2,1] * Conv_2_aggregated_features[1,14]
   + A[3,1] * Conv_2_aggregated_features[1,14]
   + A[4,1] * Conv_2_aggregated_features[1,14]
   + A[5,1] * Conv_2_aggregated_features[1,14]
   + A[6,1] * Conv_2_aggregated_features[1,14]
   + A[7,1] * Conv_2_aggregated_features[1,14]
   + A[8,1] * Conv_2_aggregated_features[1,14]
   + A[9,1] * Conv_2_aggregated_features[1,14] ] = 0
 Conv_2_averages_constraint[1,15]: [
   A[0,1] * Conv_2_aggregated_features[1,15]
   - A[1,0] * Conv_1_Relu_ts[0,15] - A[1,1] * Conv_1_Relu_ts[1,15]
   + A[1,1] * Conv_2_aggregated_features[1,15]
   - A[1,2] * Conv_1_Relu_ts[2,15] - A[1,3] * Conv_1_Relu_ts[3,15]
   - A[1,4] * Conv_1_Relu_ts[4,15] - A[1,5] * Conv_1_Relu_ts[5,15]
   - A[1,6] * Conv_1_Relu_ts[6,15] - A[1,7] * Conv_1_Relu_ts[7,15]
   - A[1,8] * Conv_1_Relu_ts[8,15] - A[1,9] * Conv_1_Relu_ts[9,15]
   + A[2,1] * Conv_2_aggregated_features[1,15]
   + A[3,1] * Conv_2_aggregated_features[1,15]
   + A[4,1] * Conv_2_aggregated_features[1,15]
   + A[5,1] * Conv_2_aggregated_features[1,15]
   + A[6,1] * Conv_2_aggregated_features[1,15]
   + A[7,1] * Conv_2_aggregated_features[1,15]
   + A[8,1] * Conv_2_aggregated_features[1,15]
   + A[9,1] * Conv_2_aggregated_features[1,15] ] = 0
 Conv_2_averages_constraint[2,0]: [
   A[0,2] * Conv_2_aggregated_features[2,0]
   + A[1,2] * Conv_2_aggregated_features[2,0]
   - A[2,0] * Conv_1_Relu_ts[0,0] - A[2,1] * Conv_1_Relu_ts[1,0]
   - A[2,2] * Conv_1_Relu_ts[2,0]
   + A[2,2] * Conv_2_aggregated_features[2,0]
   - A[2,3] * Conv_1_Relu_ts[3,0] - A[2,4] * Conv_1_Relu_ts[4,0]
   - A[2,5] * Conv_1_Relu_ts[5,0] - A[2,6] * Conv_1_Relu_ts[6,0]
   - A[2,7] * Conv_1_Relu_ts[7,0] - A[2,8] * Conv_1_Relu_ts[8,0]
   - A[2,9] * Conv_1_Relu_ts[9,0]
   + A[3,2] * Conv_2_aggregated_features[2,0]
   + A[4,2] * Conv_2_aggregated_features[2,0]
   + A[5,2] * Conv_2_aggregated_features[2,0]
   + A[6,2] * Conv_2_aggregated_features[2,0]
   + A[7,2] * Conv_2_aggregated_features[2,0]
   + A[8,2] * Conv_2_aggregated_features[2,0]
   + A[9,2] * Conv_2_aggregated_features[2,0] ] = 0
 Conv_2_averages_constraint[2,1]: [
   A[0,2] * Conv_2_aggregated_features[2,1]
   + A[1,2] * Conv_2_aggregated_features[2,1]
   - A[2,0] * Conv_1_Relu_ts[0,1] - A[2,1] * Conv_1_Relu_ts[1,1]
   - A[2,2] * Conv_1_Relu_ts[2,1]
   + A[2,2] * Conv_2_aggregated_features[2,1]
   - A[2,3] * Conv_1_Relu_ts[3,1] - A[2,4] * Conv_1_Relu_ts[4,1]
   - A[2,5] * Conv_1_Relu_ts[5,1] - A[2,6] * Conv_1_Relu_ts[6,1]
   - A[2,7] * Conv_1_Relu_ts[7,1] - A[2,8] * Conv_1_Relu_ts[8,1]
   - A[2,9] * Conv_1_Relu_ts[9,1]
   + A[3,2] * Conv_2_aggregated_features[2,1]
   + A[4,2] * Conv_2_aggregated_features[2,1]
   + A[5,2] * Conv_2_aggregated_features[2,1]
   + A[6,2] * Conv_2_aggregated_features[2,1]
   + A[7,2] * Conv_2_aggregated_features[2,1]
   + A[8,2] * Conv_2_aggregated_features[2,1]
   + A[9,2] * Conv_2_aggregated_features[2,1] ] = 0
 Conv_2_averages_constraint[2,2]: [
   A[0,2] * Conv_2_aggregated_features[2,2]
   + A[1,2] * Conv_2_aggregated_features[2,2]
   - A[2,0] * Conv_1_Relu_ts[0,2] - A[2,1] * Conv_1_Relu_ts[1,2]
   - A[2,2] * Conv_1_Relu_ts[2,2]
   + A[2,2] * Conv_2_aggregated_features[2,2]
   - A[2,3] * Conv_1_Relu_ts[3,2] - A[2,4] * Conv_1_Relu_ts[4,2]
   - A[2,5] * Conv_1_Relu_ts[5,2] - A[2,6] * Conv_1_Relu_ts[6,2]
   - A[2,7] * Conv_1_Relu_ts[7,2] - A[2,8] * Conv_1_Relu_ts[8,2]
   - A[2,9] * Conv_1_Relu_ts[9,2]
   + A[3,2] * Conv_2_aggregated_features[2,2]
   + A[4,2] * Conv_2_aggregated_features[2,2]
   + A[5,2] * Conv_2_aggregated_features[2,2]
   + A[6,2] * Conv_2_aggregated_features[2,2]
   + A[7,2] * Conv_2_aggregated_features[2,2]
   + A[8,2] * Conv_2_aggregated_features[2,2]
   + A[9,2] * Conv_2_aggregated_features[2,2] ] = 0
 Conv_2_averages_constraint[2,3]: [
   A[0,2] * Conv_2_aggregated_features[2,3]
   + A[1,2] * Conv_2_aggregated_features[2,3]
   - A[2,0] * Conv_1_Relu_ts[0,3] - A[2,1] * Conv_1_Relu_ts[1,3]
   - A[2,2] * Conv_1_Relu_ts[2,3]
   + A[2,2] * Conv_2_aggregated_features[2,3]
   - A[2,3] * Conv_1_Relu_ts[3,3] - A[2,4] * Conv_1_Relu_ts[4,3]
   - A[2,5] * Conv_1_Relu_ts[5,3] - A[2,6] * Conv_1_Relu_ts[6,3]
   - A[2,7] * Conv_1_Relu_ts[7,3] - A[2,8] * Conv_1_Relu_ts[8,3]
   - A[2,9] * Conv_1_Relu_ts[9,3]
   + A[3,2] * Conv_2_aggregated_features[2,3]
   + A[4,2] * Conv_2_aggregated_features[2,3]
   + A[5,2] * Conv_2_aggregated_features[2,3]
   + A[6,2] * Conv_2_aggregated_features[2,3]
   + A[7,2] * Conv_2_aggregated_features[2,3]
   + A[8,2] * Conv_2_aggregated_features[2,3]
   + A[9,2] * Conv_2_aggregated_features[2,3] ] = 0
 Conv_2_averages_constraint[2,4]: [
   A[0,2] * Conv_2_aggregated_features[2,4]
   + A[1,2] * Conv_2_aggregated_features[2,4]
   - A[2,0] * Conv_1_Relu_ts[0,4] - A[2,1] * Conv_1_Relu_ts[1,4]
   - A[2,2] * Conv_1_Relu_ts[2,4]
   + A[2,2] * Conv_2_aggregated_features[2,4]
   - A[2,3] * Conv_1_Relu_ts[3,4] - A[2,4] * Conv_1_Relu_ts[4,4]
   - A[2,5] * Conv_1_Relu_ts[5,4] - A[2,6] * Conv_1_Relu_ts[6,4]
   - A[2,7] * Conv_1_Relu_ts[7,4] - A[2,8] * Conv_1_Relu_ts[8,4]
   - A[2,9] * Conv_1_Relu_ts[9,4]
   + A[3,2] * Conv_2_aggregated_features[2,4]
   + A[4,2] * Conv_2_aggregated_features[2,4]
   + A[5,2] * Conv_2_aggregated_features[2,4]
   + A[6,2] * Conv_2_aggregated_features[2,4]
   + A[7,2] * Conv_2_aggregated_features[2,4]
   + A[8,2] * Conv_2_aggregated_features[2,4]
   + A[9,2] * Conv_2_aggregated_features[2,4] ] = 0
 Conv_2_averages_constraint[2,5]: [
   A[0,2] * Conv_2_aggregated_features[2,5]
   + A[1,2] * Conv_2_aggregated_features[2,5]
   - A[2,0] * Conv_1_Relu_ts[0,5] - A[2,1] * Conv_1_Relu_ts[1,5]
   - A[2,2] * Conv_1_Relu_ts[2,5]
   + A[2,2] * Conv_2_aggregated_features[2,5]
   - A[2,3] * Conv_1_Relu_ts[3,5] - A[2,4] * Conv_1_Relu_ts[4,5]
   - A[2,5] * Conv_1_Relu_ts[5,5] - A[2,6] * Conv_1_Relu_ts[6,5]
   - A[2,7] * Conv_1_Relu_ts[7,5] - A[2,8] * Conv_1_Relu_ts[8,5]
   - A[2,9] * Conv_1_Relu_ts[9,5]
   + A[3,2] * Conv_2_aggregated_features[2,5]
   + A[4,2] * Conv_2_aggregated_features[2,5]
   + A[5,2] * Conv_2_aggregated_features[2,5]
   + A[6,2] * Conv_2_aggregated_features[2,5]
   + A[7,2] * Conv_2_aggregated_features[2,5]
   + A[8,2] * Conv_2_aggregated_features[2,5]
   + A[9,2] * Conv_2_aggregated_features[2,5] ] = 0
 Conv_2_averages_constraint[2,6]: [
   A[0,2] * Conv_2_aggregated_features[2,6]
   + A[1,2] * Conv_2_aggregated_features[2,6]
   - A[2,0] * Conv_1_Relu_ts[0,6] - A[2,1] * Conv_1_Relu_ts[1,6]
   - A[2,2] * Conv_1_Relu_ts[2,6]
   + A[2,2] * Conv_2_aggregated_features[2,6]
   - A[2,3] * Conv_1_Relu_ts[3,6] - A[2,4] * Conv_1_Relu_ts[4,6]
   - A[2,5] * Conv_1_Relu_ts[5,6] - A[2,6] * Conv_1_Relu_ts[6,6]
   - A[2,7] * Conv_1_Relu_ts[7,6] - A[2,8] * Conv_1_Relu_ts[8,6]
   - A[2,9] * Conv_1_Relu_ts[9,6]
   + A[3,2] * Conv_2_aggregated_features[2,6]
   + A[4,2] * Conv_2_aggregated_features[2,6]
   + A[5,2] * Conv_2_aggregated_features[2,6]
   + A[6,2] * Conv_2_aggregated_features[2,6]
   + A[7,2] * Conv_2_aggregated_features[2,6]
   + A[8,2] * Conv_2_aggregated_features[2,6]
   + A[9,2] * Conv_2_aggregated_features[2,6] ] = 0
 Conv_2_averages_constraint[2,7]: [
   A[0,2] * Conv_2_aggregated_features[2,7]
   + A[1,2] * Conv_2_aggregated_features[2,7]
   - A[2,0] * Conv_1_Relu_ts[0,7] - A[2,1] * Conv_1_Relu_ts[1,7]
   - A[2,2] * Conv_1_Relu_ts[2,7]
   + A[2,2] * Conv_2_aggregated_features[2,7]
   - A[2,3] * Conv_1_Relu_ts[3,7] - A[2,4] * Conv_1_Relu_ts[4,7]
   - A[2,5] * Conv_1_Relu_ts[5,7] - A[2,6] * Conv_1_Relu_ts[6,7]
   - A[2,7] * Conv_1_Relu_ts[7,7] - A[2,8] * Conv_1_Relu_ts[8,7]
   - A[2,9] * Conv_1_Relu_ts[9,7]
   + A[3,2] * Conv_2_aggregated_features[2,7]
   + A[4,2] * Conv_2_aggregated_features[2,7]
   + A[5,2] * Conv_2_aggregated_features[2,7]
   + A[6,2] * Conv_2_aggregated_features[2,7]
   + A[7,2] * Conv_2_aggregated_features[2,7]
   + A[8,2] * Conv_2_aggregated_features[2,7]
   + A[9,2] * Conv_2_aggregated_features[2,7] ] = 0
 Conv_2_averages_constraint[2,8]: [
   A[0,2] * Conv_2_aggregated_features[2,8]
   + A[1,2] * Conv_2_aggregated_features[2,8]
   - A[2,0] * Conv_1_Relu_ts[0,8] - A[2,1] * Conv_1_Relu_ts[1,8]
   - A[2,2] * Conv_1_Relu_ts[2,8]
   + A[2,2] * Conv_2_aggregated_features[2,8]
   - A[2,3] * Conv_1_Relu_ts[3,8] - A[2,4] * Conv_1_Relu_ts[4,8]
   - A[2,5] * Conv_1_Relu_ts[5,8] - A[2,6] * Conv_1_Relu_ts[6,8]
   - A[2,7] * Conv_1_Relu_ts[7,8] - A[2,8] * Conv_1_Relu_ts[8,8]
   - A[2,9] * Conv_1_Relu_ts[9,8]
   + A[3,2] * Conv_2_aggregated_features[2,8]
   + A[4,2] * Conv_2_aggregated_features[2,8]
   + A[5,2] * Conv_2_aggregated_features[2,8]
   + A[6,2] * Conv_2_aggregated_features[2,8]
   + A[7,2] * Conv_2_aggregated_features[2,8]
   + A[8,2] * Conv_2_aggregated_features[2,8]
   + A[9,2] * Conv_2_aggregated_features[2,8] ] = 0
 Conv_2_averages_constraint[2,9]: [
   A[0,2] * Conv_2_aggregated_features[2,9]
   + A[1,2] * Conv_2_aggregated_features[2,9]
   - A[2,0] * Conv_1_Relu_ts[0,9] - A[2,1] * Conv_1_Relu_ts[1,9]
   - A[2,2] * Conv_1_Relu_ts[2,9]
   + A[2,2] * Conv_2_aggregated_features[2,9]
   - A[2,3] * Conv_1_Relu_ts[3,9] - A[2,4] * Conv_1_Relu_ts[4,9]
   - A[2,5] * Conv_1_Relu_ts[5,9] - A[2,6] * Conv_1_Relu_ts[6,9]
   - A[2,7] * Conv_1_Relu_ts[7,9] - A[2,8] * Conv_1_Relu_ts[8,9]
   - A[2,9] * Conv_1_Relu_ts[9,9]
   + A[3,2] * Conv_2_aggregated_features[2,9]
   + A[4,2] * Conv_2_aggregated_features[2,9]
   + A[5,2] * Conv_2_aggregated_features[2,9]
   + A[6,2] * Conv_2_aggregated_features[2,9]
   + A[7,2] * Conv_2_aggregated_features[2,9]
   + A[8,2] * Conv_2_aggregated_features[2,9]
   + A[9,2] * Conv_2_aggregated_features[2,9] ] = 0
 Conv_2_averages_constraint[2,10]: [
   A[0,2] * Conv_2_aggregated_features[2,10]
   + A[1,2] * Conv_2_aggregated_features[2,10]
   - A[2,0] * Conv_1_Relu_ts[0,10] - A[2,1] * Conv_1_Relu_ts[1,10]
   - A[2,2] * Conv_1_Relu_ts[2,10]
   + A[2,2] * Conv_2_aggregated_features[2,10]
   - A[2,3] * Conv_1_Relu_ts[3,10] - A[2,4] * Conv_1_Relu_ts[4,10]
   - A[2,5] * Conv_1_Relu_ts[5,10] - A[2,6] * Conv_1_Relu_ts[6,10]
   - A[2,7] * Conv_1_Relu_ts[7,10] - A[2,8] * Conv_1_Relu_ts[8,10]
   - A[2,9] * Conv_1_Relu_ts[9,10]
   + A[3,2] * Conv_2_aggregated_features[2,10]
   + A[4,2] * Conv_2_aggregated_features[2,10]
   + A[5,2] * Conv_2_aggregated_features[2,10]
   + A[6,2] * Conv_2_aggregated_features[2,10]
   + A[7,2] * Conv_2_aggregated_features[2,10]
   + A[8,2] * Conv_2_aggregated_features[2,10]
   + A[9,2] * Conv_2_aggregated_features[2,10] ] = 0
 Conv_2_averages_constraint[2,11]: [
   A[0,2] * Conv_2_aggregated_features[2,11]
   + A[1,2] * Conv_2_aggregated_features[2,11]
   - A[2,0] * Conv_1_Relu_ts[0,11] - A[2,1] * Conv_1_Relu_ts[1,11]
   - A[2,2] * Conv_1_Relu_ts[2,11]
   + A[2,2] * Conv_2_aggregated_features[2,11]
   - A[2,3] * Conv_1_Relu_ts[3,11] - A[2,4] * Conv_1_Relu_ts[4,11]
   - A[2,5] * Conv_1_Relu_ts[5,11] - A[2,6] * Conv_1_Relu_ts[6,11]
   - A[2,7] * Conv_1_Relu_ts[7,11] - A[2,8] * Conv_1_Relu_ts[8,11]
   - A[2,9] * Conv_1_Relu_ts[9,11]
   + A[3,2] * Conv_2_aggregated_features[2,11]
   + A[4,2] * Conv_2_aggregated_features[2,11]
   + A[5,2] * Conv_2_aggregated_features[2,11]
   + A[6,2] * Conv_2_aggregated_features[2,11]
   + A[7,2] * Conv_2_aggregated_features[2,11]
   + A[8,2] * Conv_2_aggregated_features[2,11]
   + A[9,2] * Conv_2_aggregated_features[2,11] ] = 0
 Conv_2_averages_constraint[2,12]: [
   A[0,2] * Conv_2_aggregated_features[2,12]
   + A[1,2] * Conv_2_aggregated_features[2,12]
   - A[2,0] * Conv_1_Relu_ts[0,12] - A[2,1] * Conv_1_Relu_ts[1,12]
   - A[2,2] * Conv_1_Relu_ts[2,12]
   + A[2,2] * Conv_2_aggregated_features[2,12]
   - A[2,3] * Conv_1_Relu_ts[3,12] - A[2,4] * Conv_1_Relu_ts[4,12]
   - A[2,5] * Conv_1_Relu_ts[5,12] - A[2,6] * Conv_1_Relu_ts[6,12]
   - A[2,7] * Conv_1_Relu_ts[7,12] - A[2,8] * Conv_1_Relu_ts[8,12]
   - A[2,9] * Conv_1_Relu_ts[9,12]
   + A[3,2] * Conv_2_aggregated_features[2,12]
   + A[4,2] * Conv_2_aggregated_features[2,12]
   + A[5,2] * Conv_2_aggregated_features[2,12]
   + A[6,2] * Conv_2_aggregated_features[2,12]
   + A[7,2] * Conv_2_aggregated_features[2,12]
   + A[8,2] * Conv_2_aggregated_features[2,12]
   + A[9,2] * Conv_2_aggregated_features[2,12] ] = 0
 Conv_2_averages_constraint[2,13]: [
   A[0,2] * Conv_2_aggregated_features[2,13]
   + A[1,2] * Conv_2_aggregated_features[2,13]
   - A[2,0] * Conv_1_Relu_ts[0,13] - A[2,1] * Conv_1_Relu_ts[1,13]
   - A[2,2] * Conv_1_Relu_ts[2,13]
   + A[2,2] * Conv_2_aggregated_features[2,13]
   - A[2,3] * Conv_1_Relu_ts[3,13] - A[2,4] * Conv_1_Relu_ts[4,13]
   - A[2,5] * Conv_1_Relu_ts[5,13] - A[2,6] * Conv_1_Relu_ts[6,13]
   - A[2,7] * Conv_1_Relu_ts[7,13] - A[2,8] * Conv_1_Relu_ts[8,13]
   - A[2,9] * Conv_1_Relu_ts[9,13]
   + A[3,2] * Conv_2_aggregated_features[2,13]
   + A[4,2] * Conv_2_aggregated_features[2,13]
   + A[5,2] * Conv_2_aggregated_features[2,13]
   + A[6,2] * Conv_2_aggregated_features[2,13]
   + A[7,2] * Conv_2_aggregated_features[2,13]
   + A[8,2] * Conv_2_aggregated_features[2,13]
   + A[9,2] * Conv_2_aggregated_features[2,13] ] = 0
 Conv_2_averages_constraint[2,14]: [
   A[0,2] * Conv_2_aggregated_features[2,14]
   + A[1,2] * Conv_2_aggregated_features[2,14]
   - A[2,0] * Conv_1_Relu_ts[0,14] - A[2,1] * Conv_1_Relu_ts[1,14]
   - A[2,2] * Conv_1_Relu_ts[2,14]
   + A[2,2] * Conv_2_aggregated_features[2,14]
   - A[2,3] * Conv_1_Relu_ts[3,14] - A[2,4] * Conv_1_Relu_ts[4,14]
   - A[2,5] * Conv_1_Relu_ts[5,14] - A[2,6] * Conv_1_Relu_ts[6,14]
   - A[2,7] * Conv_1_Relu_ts[7,14] - A[2,8] * Conv_1_Relu_ts[8,14]
   - A[2,9] * Conv_1_Relu_ts[9,14]
   + A[3,2] * Conv_2_aggregated_features[2,14]
   + A[4,2] * Conv_2_aggregated_features[2,14]
   + A[5,2] * Conv_2_aggregated_features[2,14]
   + A[6,2] * Conv_2_aggregated_features[2,14]
   + A[7,2] * Conv_2_aggregated_features[2,14]
   + A[8,2] * Conv_2_aggregated_features[2,14]
   + A[9,2] * Conv_2_aggregated_features[2,14] ] = 0
 Conv_2_averages_constraint[2,15]: [
   A[0,2] * Conv_2_aggregated_features[2,15]
   + A[1,2] * Conv_2_aggregated_features[2,15]
   - A[2,0] * Conv_1_Relu_ts[0,15] - A[2,1] * Conv_1_Relu_ts[1,15]
   - A[2,2] * Conv_1_Relu_ts[2,15]
   + A[2,2] * Conv_2_aggregated_features[2,15]
   - A[2,3] * Conv_1_Relu_ts[3,15] - A[2,4] * Conv_1_Relu_ts[4,15]
   - A[2,5] * Conv_1_Relu_ts[5,15] - A[2,6] * Conv_1_Relu_ts[6,15]
   - A[2,7] * Conv_1_Relu_ts[7,15] - A[2,8] * Conv_1_Relu_ts[8,15]
   - A[2,9] * Conv_1_Relu_ts[9,15]
   + A[3,2] * Conv_2_aggregated_features[2,15]
   + A[4,2] * Conv_2_aggregated_features[2,15]
   + A[5,2] * Conv_2_aggregated_features[2,15]
   + A[6,2] * Conv_2_aggregated_features[2,15]
   + A[7,2] * Conv_2_aggregated_features[2,15]
   + A[8,2] * Conv_2_aggregated_features[2,15]
   + A[9,2] * Conv_2_aggregated_features[2,15] ] = 0
 Conv_2_averages_constraint[3,0]: [
   A[0,3] * Conv_2_aggregated_features[3,0]
   + A[1,3] * Conv_2_aggregated_features[3,0]
   + A[2,3] * Conv_2_aggregated_features[3,0]
   - A[3,0] * Conv_1_Relu_ts[0,0] - A[3,1] * Conv_1_Relu_ts[1,0]
   - A[3,2] * Conv_1_Relu_ts[2,0] - A[3,3] * Conv_1_Relu_ts[3,0]
   + A[3,3] * Conv_2_aggregated_features[3,0]
   - A[3,4] * Conv_1_Relu_ts[4,0] - A[3,5] * Conv_1_Relu_ts[5,0]
   - A[3,6] * Conv_1_Relu_ts[6,0] - A[3,7] * Conv_1_Relu_ts[7,0]
   - A[3,8] * Conv_1_Relu_ts[8,0] - A[3,9] * Conv_1_Relu_ts[9,0]
   + A[4,3] * Conv_2_aggregated_features[3,0]
   + A[5,3] * Conv_2_aggregated_features[3,0]
   + A[6,3] * Conv_2_aggregated_features[3,0]
   + A[7,3] * Conv_2_aggregated_features[3,0]
   + A[8,3] * Conv_2_aggregated_features[3,0]
   + A[9,3] * Conv_2_aggregated_features[3,0] ] = 0
 Conv_2_averages_constraint[3,1]: [
   A[0,3] * Conv_2_aggregated_features[3,1]
   + A[1,3] * Conv_2_aggregated_features[3,1]
   + A[2,3] * Conv_2_aggregated_features[3,1]
   - A[3,0] * Conv_1_Relu_ts[0,1] - A[3,1] * Conv_1_Relu_ts[1,1]
   - A[3,2] * Conv_1_Relu_ts[2,1] - A[3,3] * Conv_1_Relu_ts[3,1]
   + A[3,3] * Conv_2_aggregated_features[3,1]
   - A[3,4] * Conv_1_Relu_ts[4,1] - A[3,5] * Conv_1_Relu_ts[5,1]
   - A[3,6] * Conv_1_Relu_ts[6,1] - A[3,7] * Conv_1_Relu_ts[7,1]
   - A[3,8] * Conv_1_Relu_ts[8,1] - A[3,9] * Conv_1_Relu_ts[9,1]
   + A[4,3] * Conv_2_aggregated_features[3,1]
   + A[5,3] * Conv_2_aggregated_features[3,1]
   + A[6,3] * Conv_2_aggregated_features[3,1]
   + A[7,3] * Conv_2_aggregated_features[3,1]
   + A[8,3] * Conv_2_aggregated_features[3,1]
   + A[9,3] * Conv_2_aggregated_features[3,1] ] = 0
 Conv_2_averages_constraint[3,2]: [
   A[0,3] * Conv_2_aggregated_features[3,2]
   + A[1,3] * Conv_2_aggregated_features[3,2]
   + A[2,3] * Conv_2_aggregated_features[3,2]
   - A[3,0] * Conv_1_Relu_ts[0,2] - A[3,1] * Conv_1_Relu_ts[1,2]
   - A[3,2] * Conv_1_Relu_ts[2,2] - A[3,3] * Conv_1_Relu_ts[3,2]
   + A[3,3] * Conv_2_aggregated_features[3,2]
   - A[3,4] * Conv_1_Relu_ts[4,2] - A[3,5] * Conv_1_Relu_ts[5,2]
   - A[3,6] * Conv_1_Relu_ts[6,2] - A[3,7] * Conv_1_Relu_ts[7,2]
   - A[3,8] * Conv_1_Relu_ts[8,2] - A[3,9] * Conv_1_Relu_ts[9,2]
   + A[4,3] * Conv_2_aggregated_features[3,2]
   + A[5,3] * Conv_2_aggregated_features[3,2]
   + A[6,3] * Conv_2_aggregated_features[3,2]
   + A[7,3] * Conv_2_aggregated_features[3,2]
   + A[8,3] * Conv_2_aggregated_features[3,2]
   + A[9,3] * Conv_2_aggregated_features[3,2] ] = 0
 Conv_2_averages_constraint[3,3]: [
   A[0,3] * Conv_2_aggregated_features[3,3]
   + A[1,3] * Conv_2_aggregated_features[3,3]
   + A[2,3] * Conv_2_aggregated_features[3,3]
   - A[3,0] * Conv_1_Relu_ts[0,3] - A[3,1] * Conv_1_Relu_ts[1,3]
   - A[3,2] * Conv_1_Relu_ts[2,3] - A[3,3] * Conv_1_Relu_ts[3,3]
   + A[3,3] * Conv_2_aggregated_features[3,3]
   - A[3,4] * Conv_1_Relu_ts[4,3] - A[3,5] * Conv_1_Relu_ts[5,3]
   - A[3,6] * Conv_1_Relu_ts[6,3] - A[3,7] * Conv_1_Relu_ts[7,3]
   - A[3,8] * Conv_1_Relu_ts[8,3] - A[3,9] * Conv_1_Relu_ts[9,3]
   + A[4,3] * Conv_2_aggregated_features[3,3]
   + A[5,3] * Conv_2_aggregated_features[3,3]
   + A[6,3] * Conv_2_aggregated_features[3,3]
   + A[7,3] * Conv_2_aggregated_features[3,3]
   + A[8,3] * Conv_2_aggregated_features[3,3]
   + A[9,3] * Conv_2_aggregated_features[3,3] ] = 0
 Conv_2_averages_constraint[3,4]: [
   A[0,3] * Conv_2_aggregated_features[3,4]
   + A[1,3] * Conv_2_aggregated_features[3,4]
   + A[2,3] * Conv_2_aggregated_features[3,4]
   - A[3,0] * Conv_1_Relu_ts[0,4] - A[3,1] * Conv_1_Relu_ts[1,4]
   - A[3,2] * Conv_1_Relu_ts[2,4] - A[3,3] * Conv_1_Relu_ts[3,4]
   + A[3,3] * Conv_2_aggregated_features[3,4]
   - A[3,4] * Conv_1_Relu_ts[4,4] - A[3,5] * Conv_1_Relu_ts[5,4]
   - A[3,6] * Conv_1_Relu_ts[6,4] - A[3,7] * Conv_1_Relu_ts[7,4]
   - A[3,8] * Conv_1_Relu_ts[8,4] - A[3,9] * Conv_1_Relu_ts[9,4]
   + A[4,3] * Conv_2_aggregated_features[3,4]
   + A[5,3] * Conv_2_aggregated_features[3,4]
   + A[6,3] * Conv_2_aggregated_features[3,4]
   + A[7,3] * Conv_2_aggregated_features[3,4]
   + A[8,3] * Conv_2_aggregated_features[3,4]
   + A[9,3] * Conv_2_aggregated_features[3,4] ] = 0
 Conv_2_averages_constraint[3,5]: [
   A[0,3] * Conv_2_aggregated_features[3,5]
   + A[1,3] * Conv_2_aggregated_features[3,5]
   + A[2,3] * Conv_2_aggregated_features[3,5]
   - A[3,0] * Conv_1_Relu_ts[0,5] - A[3,1] * Conv_1_Relu_ts[1,5]
   - A[3,2] * Conv_1_Relu_ts[2,5] - A[3,3] * Conv_1_Relu_ts[3,5]
   + A[3,3] * Conv_2_aggregated_features[3,5]
   - A[3,4] * Conv_1_Relu_ts[4,5] - A[3,5] * Conv_1_Relu_ts[5,5]
   - A[3,6] * Conv_1_Relu_ts[6,5] - A[3,7] * Conv_1_Relu_ts[7,5]
   - A[3,8] * Conv_1_Relu_ts[8,5] - A[3,9] * Conv_1_Relu_ts[9,5]
   + A[4,3] * Conv_2_aggregated_features[3,5]
   + A[5,3] * Conv_2_aggregated_features[3,5]
   + A[6,3] * Conv_2_aggregated_features[3,5]
   + A[7,3] * Conv_2_aggregated_features[3,5]
   + A[8,3] * Conv_2_aggregated_features[3,5]
   + A[9,3] * Conv_2_aggregated_features[3,5] ] = 0
 Conv_2_averages_constraint[3,6]: [
   A[0,3] * Conv_2_aggregated_features[3,6]
   + A[1,3] * Conv_2_aggregated_features[3,6]
   + A[2,3] * Conv_2_aggregated_features[3,6]
   - A[3,0] * Conv_1_Relu_ts[0,6] - A[3,1] * Conv_1_Relu_ts[1,6]
   - A[3,2] * Conv_1_Relu_ts[2,6] - A[3,3] * Conv_1_Relu_ts[3,6]
   + A[3,3] * Conv_2_aggregated_features[3,6]
   - A[3,4] * Conv_1_Relu_ts[4,6] - A[3,5] * Conv_1_Relu_ts[5,6]
   - A[3,6] * Conv_1_Relu_ts[6,6] - A[3,7] * Conv_1_Relu_ts[7,6]
   - A[3,8] * Conv_1_Relu_ts[8,6] - A[3,9] * Conv_1_Relu_ts[9,6]
   + A[4,3] * Conv_2_aggregated_features[3,6]
   + A[5,3] * Conv_2_aggregated_features[3,6]
   + A[6,3] * Conv_2_aggregated_features[3,6]
   + A[7,3] * Conv_2_aggregated_features[3,6]
   + A[8,3] * Conv_2_aggregated_features[3,6]
   + A[9,3] * Conv_2_aggregated_features[3,6] ] = 0
 Conv_2_averages_constraint[3,7]: [
   A[0,3] * Conv_2_aggregated_features[3,7]
   + A[1,3] * Conv_2_aggregated_features[3,7]
   + A[2,3] * Conv_2_aggregated_features[3,7]
   - A[3,0] * Conv_1_Relu_ts[0,7] - A[3,1] * Conv_1_Relu_ts[1,7]
   - A[3,2] * Conv_1_Relu_ts[2,7] - A[3,3] * Conv_1_Relu_ts[3,7]
   + A[3,3] * Conv_2_aggregated_features[3,7]
   - A[3,4] * Conv_1_Relu_ts[4,7] - A[3,5] * Conv_1_Relu_ts[5,7]
   - A[3,6] * Conv_1_Relu_ts[6,7] - A[3,7] * Conv_1_Relu_ts[7,7]
   - A[3,8] * Conv_1_Relu_ts[8,7] - A[3,9] * Conv_1_Relu_ts[9,7]
   + A[4,3] * Conv_2_aggregated_features[3,7]
   + A[5,3] * Conv_2_aggregated_features[3,7]
   + A[6,3] * Conv_2_aggregated_features[3,7]
   + A[7,3] * Conv_2_aggregated_features[3,7]
   + A[8,3] * Conv_2_aggregated_features[3,7]
   + A[9,3] * Conv_2_aggregated_features[3,7] ] = 0
 Conv_2_averages_constraint[3,8]: [
   A[0,3] * Conv_2_aggregated_features[3,8]
   + A[1,3] * Conv_2_aggregated_features[3,8]
   + A[2,3] * Conv_2_aggregated_features[3,8]
   - A[3,0] * Conv_1_Relu_ts[0,8] - A[3,1] * Conv_1_Relu_ts[1,8]
   - A[3,2] * Conv_1_Relu_ts[2,8] - A[3,3] * Conv_1_Relu_ts[3,8]
   + A[3,3] * Conv_2_aggregated_features[3,8]
   - A[3,4] * Conv_1_Relu_ts[4,8] - A[3,5] * Conv_1_Relu_ts[5,8]
   - A[3,6] * Conv_1_Relu_ts[6,8] - A[3,7] * Conv_1_Relu_ts[7,8]
   - A[3,8] * Conv_1_Relu_ts[8,8] - A[3,9] * Conv_1_Relu_ts[9,8]
   + A[4,3] * Conv_2_aggregated_features[3,8]
   + A[5,3] * Conv_2_aggregated_features[3,8]
   + A[6,3] * Conv_2_aggregated_features[3,8]
   + A[7,3] * Conv_2_aggregated_features[3,8]
   + A[8,3] * Conv_2_aggregated_features[3,8]
   + A[9,3] * Conv_2_aggregated_features[3,8] ] = 0
 Conv_2_averages_constraint[3,9]: [
   A[0,3] * Conv_2_aggregated_features[3,9]
   + A[1,3] * Conv_2_aggregated_features[3,9]
   + A[2,3] * Conv_2_aggregated_features[3,9]
   - A[3,0] * Conv_1_Relu_ts[0,9] - A[3,1] * Conv_1_Relu_ts[1,9]
   - A[3,2] * Conv_1_Relu_ts[2,9] - A[3,3] * Conv_1_Relu_ts[3,9]
   + A[3,3] * Conv_2_aggregated_features[3,9]
   - A[3,4] * Conv_1_Relu_ts[4,9] - A[3,5] * Conv_1_Relu_ts[5,9]
   - A[3,6] * Conv_1_Relu_ts[6,9] - A[3,7] * Conv_1_Relu_ts[7,9]
   - A[3,8] * Conv_1_Relu_ts[8,9] - A[3,9] * Conv_1_Relu_ts[9,9]
   + A[4,3] * Conv_2_aggregated_features[3,9]
   + A[5,3] * Conv_2_aggregated_features[3,9]
   + A[6,3] * Conv_2_aggregated_features[3,9]
   + A[7,3] * Conv_2_aggregated_features[3,9]
   + A[8,3] * Conv_2_aggregated_features[3,9]
   + A[9,3] * Conv_2_aggregated_features[3,9] ] = 0
 Conv_2_averages_constraint[3,10]: [
   A[0,3] * Conv_2_aggregated_features[3,10]
   + A[1,3] * Conv_2_aggregated_features[3,10]
   + A[2,3] * Conv_2_aggregated_features[3,10]
   - A[3,0] * Conv_1_Relu_ts[0,10] - A[3,1] * Conv_1_Relu_ts[1,10]
   - A[3,2] * Conv_1_Relu_ts[2,10] - A[3,3] * Conv_1_Relu_ts[3,10]
   + A[3,3] * Conv_2_aggregated_features[3,10]
   - A[3,4] * Conv_1_Relu_ts[4,10] - A[3,5] * Conv_1_Relu_ts[5,10]
   - A[3,6] * Conv_1_Relu_ts[6,10] - A[3,7] * Conv_1_Relu_ts[7,10]
   - A[3,8] * Conv_1_Relu_ts[8,10] - A[3,9] * Conv_1_Relu_ts[9,10]
   + A[4,3] * Conv_2_aggregated_features[3,10]
   + A[5,3] * Conv_2_aggregated_features[3,10]
   + A[6,3] * Conv_2_aggregated_features[3,10]
   + A[7,3] * Conv_2_aggregated_features[3,10]
   + A[8,3] * Conv_2_aggregated_features[3,10]
   + A[9,3] * Conv_2_aggregated_features[3,10] ] = 0
 Conv_2_averages_constraint[3,11]: [
   A[0,3] * Conv_2_aggregated_features[3,11]
   + A[1,3] * Conv_2_aggregated_features[3,11]
   + A[2,3] * Conv_2_aggregated_features[3,11]
   - A[3,0] * Conv_1_Relu_ts[0,11] - A[3,1] * Conv_1_Relu_ts[1,11]
   - A[3,2] * Conv_1_Relu_ts[2,11] - A[3,3] * Conv_1_Relu_ts[3,11]
   + A[3,3] * Conv_2_aggregated_features[3,11]
   - A[3,4] * Conv_1_Relu_ts[4,11] - A[3,5] * Conv_1_Relu_ts[5,11]
   - A[3,6] * Conv_1_Relu_ts[6,11] - A[3,7] * Conv_1_Relu_ts[7,11]
   - A[3,8] * Conv_1_Relu_ts[8,11] - A[3,9] * Conv_1_Relu_ts[9,11]
   + A[4,3] * Conv_2_aggregated_features[3,11]
   + A[5,3] * Conv_2_aggregated_features[3,11]
   + A[6,3] * Conv_2_aggregated_features[3,11]
   + A[7,3] * Conv_2_aggregated_features[3,11]
   + A[8,3] * Conv_2_aggregated_features[3,11]
   + A[9,3] * Conv_2_aggregated_features[3,11] ] = 0
 Conv_2_averages_constraint[3,12]: [
   A[0,3] * Conv_2_aggregated_features[3,12]
   + A[1,3] * Conv_2_aggregated_features[3,12]
   + A[2,3] * Conv_2_aggregated_features[3,12]
   - A[3,0] * Conv_1_Relu_ts[0,12] - A[3,1] * Conv_1_Relu_ts[1,12]
   - A[3,2] * Conv_1_Relu_ts[2,12] - A[3,3] * Conv_1_Relu_ts[3,12]
   + A[3,3] * Conv_2_aggregated_features[3,12]
   - A[3,4] * Conv_1_Relu_ts[4,12] - A[3,5] * Conv_1_Relu_ts[5,12]
   - A[3,6] * Conv_1_Relu_ts[6,12] - A[3,7] * Conv_1_Relu_ts[7,12]
   - A[3,8] * Conv_1_Relu_ts[8,12] - A[3,9] * Conv_1_Relu_ts[9,12]
   + A[4,3] * Conv_2_aggregated_features[3,12]
   + A[5,3] * Conv_2_aggregated_features[3,12]
   + A[6,3] * Conv_2_aggregated_features[3,12]
   + A[7,3] * Conv_2_aggregated_features[3,12]
   + A[8,3] * Conv_2_aggregated_features[3,12]
   + A[9,3] * Conv_2_aggregated_features[3,12] ] = 0
 Conv_2_averages_constraint[3,13]: [
   A[0,3] * Conv_2_aggregated_features[3,13]
   + A[1,3] * Conv_2_aggregated_features[3,13]
   + A[2,3] * Conv_2_aggregated_features[3,13]
   - A[3,0] * Conv_1_Relu_ts[0,13] - A[3,1] * Conv_1_Relu_ts[1,13]
   - A[3,2] * Conv_1_Relu_ts[2,13] - A[3,3] * Conv_1_Relu_ts[3,13]
   + A[3,3] * Conv_2_aggregated_features[3,13]
   - A[3,4] * Conv_1_Relu_ts[4,13] - A[3,5] * Conv_1_Relu_ts[5,13]
   - A[3,6] * Conv_1_Relu_ts[6,13] - A[3,7] * Conv_1_Relu_ts[7,13]
   - A[3,8] * Conv_1_Relu_ts[8,13] - A[3,9] * Conv_1_Relu_ts[9,13]
   + A[4,3] * Conv_2_aggregated_features[3,13]
   + A[5,3] * Conv_2_aggregated_features[3,13]
   + A[6,3] * Conv_2_aggregated_features[3,13]
   + A[7,3] * Conv_2_aggregated_features[3,13]
   + A[8,3] * Conv_2_aggregated_features[3,13]
   + A[9,3] * Conv_2_aggregated_features[3,13] ] = 0
 Conv_2_averages_constraint[3,14]: [
   A[0,3] * Conv_2_aggregated_features[3,14]
   + A[1,3] * Conv_2_aggregated_features[3,14]
   + A[2,3] * Conv_2_aggregated_features[3,14]
   - A[3,0] * Conv_1_Relu_ts[0,14] - A[3,1] * Conv_1_Relu_ts[1,14]
   - A[3,2] * Conv_1_Relu_ts[2,14] - A[3,3] * Conv_1_Relu_ts[3,14]
   + A[3,3] * Conv_2_aggregated_features[3,14]
   - A[3,4] * Conv_1_Relu_ts[4,14] - A[3,5] * Conv_1_Relu_ts[5,14]
   - A[3,6] * Conv_1_Relu_ts[6,14] - A[3,7] * Conv_1_Relu_ts[7,14]
   - A[3,8] * Conv_1_Relu_ts[8,14] - A[3,9] * Conv_1_Relu_ts[9,14]
   + A[4,3] * Conv_2_aggregated_features[3,14]
   + A[5,3] * Conv_2_aggregated_features[3,14]
   + A[6,3] * Conv_2_aggregated_features[3,14]
   + A[7,3] * Conv_2_aggregated_features[3,14]
   + A[8,3] * Conv_2_aggregated_features[3,14]
   + A[9,3] * Conv_2_aggregated_features[3,14] ] = 0
 Conv_2_averages_constraint[3,15]: [
   A[0,3] * Conv_2_aggregated_features[3,15]
   + A[1,3] * Conv_2_aggregated_features[3,15]
   + A[2,3] * Conv_2_aggregated_features[3,15]
   - A[3,0] * Conv_1_Relu_ts[0,15] - A[3,1] * Conv_1_Relu_ts[1,15]
   - A[3,2] * Conv_1_Relu_ts[2,15] - A[3,3] * Conv_1_Relu_ts[3,15]
   + A[3,3] * Conv_2_aggregated_features[3,15]
   - A[3,4] * Conv_1_Relu_ts[4,15] - A[3,5] * Conv_1_Relu_ts[5,15]
   - A[3,6] * Conv_1_Relu_ts[6,15] - A[3,7] * Conv_1_Relu_ts[7,15]
   - A[3,8] * Conv_1_Relu_ts[8,15] - A[3,9] * Conv_1_Relu_ts[9,15]
   + A[4,3] * Conv_2_aggregated_features[3,15]
   + A[5,3] * Conv_2_aggregated_features[3,15]
   + A[6,3] * Conv_2_aggregated_features[3,15]
   + A[7,3] * Conv_2_aggregated_features[3,15]
   + A[8,3] * Conv_2_aggregated_features[3,15]
   + A[9,3] * Conv_2_aggregated_features[3,15] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,0] - A[4,8] * Conv_1_Relu_ts[8,0]
   - A[4,9] * Conv_1_Relu_ts[9,0]
   + A[5,4] * Conv_2_aggregated_features[4,0]
   + A[6,4] * Conv_2_aggregated_features[4,0]
   + A[7,4] * Conv_2_aggregated_features[4,0]
   + A[8,4] * Conv_2_aggregated_features[4,0]
   + A[9,4] * Conv_2_aggregated_features[4,0] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,1] - A[4,8] * Conv_1_Relu_ts[8,1]
   - A[4,9] * Conv_1_Relu_ts[9,1]
   + A[5,4] * Conv_2_aggregated_features[4,1]
   + A[6,4] * Conv_2_aggregated_features[4,1]
   + A[7,4] * Conv_2_aggregated_features[4,1]
   + A[8,4] * Conv_2_aggregated_features[4,1]
   + A[9,4] * Conv_2_aggregated_features[4,1] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,2] - A[4,8] * Conv_1_Relu_ts[8,2]
   - A[4,9] * Conv_1_Relu_ts[9,2]
   + A[5,4] * Conv_2_aggregated_features[4,2]
   + A[6,4] * Conv_2_aggregated_features[4,2]
   + A[7,4] * Conv_2_aggregated_features[4,2]
   + A[8,4] * Conv_2_aggregated_features[4,2]
   + A[9,4] * Conv_2_aggregated_features[4,2] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,3] - A[4,8] * Conv_1_Relu_ts[8,3]
   - A[4,9] * Conv_1_Relu_ts[9,3]
   + A[5,4] * Conv_2_aggregated_features[4,3]
   + A[6,4] * Conv_2_aggregated_features[4,3]
   + A[7,4] * Conv_2_aggregated_features[4,3]
   + A[8,4] * Conv_2_aggregated_features[4,3]
   + A[9,4] * Conv_2_aggregated_features[4,3] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,4] - A[4,8] * Conv_1_Relu_ts[8,4]
   - A[4,9] * Conv_1_Relu_ts[9,4]
   + A[5,4] * Conv_2_aggregated_features[4,4]
   + A[6,4] * Conv_2_aggregated_features[4,4]
   + A[7,4] * Conv_2_aggregated_features[4,4]
   + A[8,4] * Conv_2_aggregated_features[4,4]
   + A[9,4] * Conv_2_aggregated_features[4,4] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,5] - A[4,8] * Conv_1_Relu_ts[8,5]
   - A[4,9] * Conv_1_Relu_ts[9,5]
   + A[5,4] * Conv_2_aggregated_features[4,5]
   + A[6,4] * Conv_2_aggregated_features[4,5]
   + A[7,4] * Conv_2_aggregated_features[4,5]
   + A[8,4] * Conv_2_aggregated_features[4,5]
   + A[9,4] * Conv_2_aggregated_features[4,5] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,6] - A[4,8] * Conv_1_Relu_ts[8,6]
   - A[4,9] * Conv_1_Relu_ts[9,6]
   + A[5,4] * Conv_2_aggregated_features[4,6]
   + A[6,4] * Conv_2_aggregated_features[4,6]
   + A[7,4] * Conv_2_aggregated_features[4,6]
   + A[8,4] * Conv_2_aggregated_features[4,6]
   + A[9,4] * Conv_2_aggregated_features[4,6] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,7] - A[4,8] * Conv_1_Relu_ts[8,7]
   - A[4,9] * Conv_1_Relu_ts[9,7]
   + A[5,4] * Conv_2_aggregated_features[4,7]
   + A[6,4] * Conv_2_aggregated_features[4,7]
   + A[7,4] * Conv_2_aggregated_features[4,7]
   + A[8,4] * Conv_2_aggregated_features[4,7]
   + A[9,4] * Conv_2_aggregated_features[4,7] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,8] - A[4,8] * Conv_1_Relu_ts[8,8]
   - A[4,9] * Conv_1_Relu_ts[9,8]
   + A[5,4] * Conv_2_aggregated_features[4,8]
   + A[6,4] * Conv_2_aggregated_features[4,8]
   + A[7,4] * Conv_2_aggregated_features[4,8]
   + A[8,4] * Conv_2_aggregated_features[4,8]
   + A[9,4] * Conv_2_aggregated_features[4,8] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,9] - A[4,8] * Conv_1_Relu_ts[8,9]
   - A[4,9] * Conv_1_Relu_ts[9,9]
   + A[5,4] * Conv_2_aggregated_features[4,9]
   + A[6,4] * Conv_2_aggregated_features[4,9]
   + A[7,4] * Conv_2_aggregated_features[4,9]
   + A[8,4] * Conv_2_aggregated_features[4,9]
   + A[9,4] * Conv_2_aggregated_features[4,9] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,10] - A[4,8] * Conv_1_Relu_ts[8,10]
   - A[4,9] * Conv_1_Relu_ts[9,10]
   + A[5,4] * Conv_2_aggregated_features[4,10]
   + A[6,4] * Conv_2_aggregated_features[4,10]
   + A[7,4] * Conv_2_aggregated_features[4,10]
   + A[8,4] * Conv_2_aggregated_features[4,10]
   + A[9,4] * Conv_2_aggregated_features[4,10] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,11] - A[4,8] * Conv_1_Relu_ts[8,11]
   - A[4,9] * Conv_1_Relu_ts[9,11]
   + A[5,4] * Conv_2_aggregated_features[4,11]
   + A[6,4] * Conv_2_aggregated_features[4,11]
   + A[7,4] * Conv_2_aggregated_features[4,11]
   + A[8,4] * Conv_2_aggregated_features[4,11]
   + A[9,4] * Conv_2_aggregated_features[4,11] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,12] - A[4,8] * Conv_1_Relu_ts[8,12]
   - A[4,9] * Conv_1_Relu_ts[9,12]
   + A[5,4] * Conv_2_aggregated_features[4,12]
   + A[6,4] * Conv_2_aggregated_features[4,12]
   + A[7,4] * Conv_2_aggregated_features[4,12]
   + A[8,4] * Conv_2_aggregated_features[4,12]
   + A[9,4] * Conv_2_aggregated_features[4,12] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,13] - A[4,8] * Conv_1_Relu_ts[8,13]
   - A[4,9] * Conv_1_Relu_ts[9,13]
   + A[5,4] * Conv_2_aggregated_features[4,13]
   + A[6,4] * Conv_2_aggregated_features[4,13]
   + A[7,4] * Conv_2_aggregated_features[4,13]
   + A[8,4] * Conv_2_aggregated_features[4,13]
   + A[9,4] * Conv_2_aggregated_features[4,13] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,14] - A[4,8] * Conv_1_Relu_ts[8,14]
   - A[4,9] * Conv_1_Relu_ts[9,14]
   + A[5,4] * Conv_2_aggregated_features[4,14]
   + A[6,4] * Conv_2_aggregated_features[4,14]
   + A[7,4] * Conv_2_aggregated_features[4,14]
   + A[8,4] * Conv_2_aggregated_features[4,14]
   + A[9,4] * Conv_2_aggregated_features[4,14] ] = 0
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
   - A[4,7] * Conv_1_Relu_ts[7,15] - A[4,8] * Conv_1_Relu_ts[8,15]
   - A[4,9] * Conv_1_Relu_ts[9,15]
   + A[5,4] * Conv_2_aggregated_features[4,15]
   + A[6,4] * Conv_2_aggregated_features[4,15]
   + A[7,4] * Conv_2_aggregated_features[4,15]
   + A[8,4] * Conv_2_aggregated_features[4,15]
   + A[9,4] * Conv_2_aggregated_features[4,15] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,0] - A[5,9] * Conv_1_Relu_ts[9,0]
   + A[6,5] * Conv_2_aggregated_features[5,0]
   + A[7,5] * Conv_2_aggregated_features[5,0]
   + A[8,5] * Conv_2_aggregated_features[5,0]
   + A[9,5] * Conv_2_aggregated_features[5,0] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,1] - A[5,9] * Conv_1_Relu_ts[9,1]
   + A[6,5] * Conv_2_aggregated_features[5,1]
   + A[7,5] * Conv_2_aggregated_features[5,1]
   + A[8,5] * Conv_2_aggregated_features[5,1]
   + A[9,5] * Conv_2_aggregated_features[5,1] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,2] - A[5,9] * Conv_1_Relu_ts[9,2]
   + A[6,5] * Conv_2_aggregated_features[5,2]
   + A[7,5] * Conv_2_aggregated_features[5,2]
   + A[8,5] * Conv_2_aggregated_features[5,2]
   + A[9,5] * Conv_2_aggregated_features[5,2] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,3] - A[5,9] * Conv_1_Relu_ts[9,3]
   + A[6,5] * Conv_2_aggregated_features[5,3]
   + A[7,5] * Conv_2_aggregated_features[5,3]
   + A[8,5] * Conv_2_aggregated_features[5,3]
   + A[9,5] * Conv_2_aggregated_features[5,3] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,4] - A[5,9] * Conv_1_Relu_ts[9,4]
   + A[6,5] * Conv_2_aggregated_features[5,4]
   + A[7,5] * Conv_2_aggregated_features[5,4]
   + A[8,5] * Conv_2_aggregated_features[5,4]
   + A[9,5] * Conv_2_aggregated_features[5,4] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,5] - A[5,9] * Conv_1_Relu_ts[9,5]
   + A[6,5] * Conv_2_aggregated_features[5,5]
   + A[7,5] * Conv_2_aggregated_features[5,5]
   + A[8,5] * Conv_2_aggregated_features[5,5]
   + A[9,5] * Conv_2_aggregated_features[5,5] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,6] - A[5,9] * Conv_1_Relu_ts[9,6]
   + A[6,5] * Conv_2_aggregated_features[5,6]
   + A[7,5] * Conv_2_aggregated_features[5,6]
   + A[8,5] * Conv_2_aggregated_features[5,6]
   + A[9,5] * Conv_2_aggregated_features[5,6] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,7] - A[5,9] * Conv_1_Relu_ts[9,7]
   + A[6,5] * Conv_2_aggregated_features[5,7]
   + A[7,5] * Conv_2_aggregated_features[5,7]
   + A[8,5] * Conv_2_aggregated_features[5,7]
   + A[9,5] * Conv_2_aggregated_features[5,7] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,8] - A[5,9] * Conv_1_Relu_ts[9,8]
   + A[6,5] * Conv_2_aggregated_features[5,8]
   + A[7,5] * Conv_2_aggregated_features[5,8]
   + A[8,5] * Conv_2_aggregated_features[5,8]
   + A[9,5] * Conv_2_aggregated_features[5,8] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,9] - A[5,9] * Conv_1_Relu_ts[9,9]
   + A[6,5] * Conv_2_aggregated_features[5,9]
   + A[7,5] * Conv_2_aggregated_features[5,9]
   + A[8,5] * Conv_2_aggregated_features[5,9]
   + A[9,5] * Conv_2_aggregated_features[5,9] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,10] - A[5,9] * Conv_1_Relu_ts[9,10]
   + A[6,5] * Conv_2_aggregated_features[5,10]
   + A[7,5] * Conv_2_aggregated_features[5,10]
   + A[8,5] * Conv_2_aggregated_features[5,10]
   + A[9,5] * Conv_2_aggregated_features[5,10] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,11] - A[5,9] * Conv_1_Relu_ts[9,11]
   + A[6,5] * Conv_2_aggregated_features[5,11]
   + A[7,5] * Conv_2_aggregated_features[5,11]
   + A[8,5] * Conv_2_aggregated_features[5,11]
   + A[9,5] * Conv_2_aggregated_features[5,11] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,12] - A[5,9] * Conv_1_Relu_ts[9,12]
   + A[6,5] * Conv_2_aggregated_features[5,12]
   + A[7,5] * Conv_2_aggregated_features[5,12]
   + A[8,5] * Conv_2_aggregated_features[5,12]
   + A[9,5] * Conv_2_aggregated_features[5,12] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,13] - A[5,9] * Conv_1_Relu_ts[9,13]
   + A[6,5] * Conv_2_aggregated_features[5,13]
   + A[7,5] * Conv_2_aggregated_features[5,13]
   + A[8,5] * Conv_2_aggregated_features[5,13]
   + A[9,5] * Conv_2_aggregated_features[5,13] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,14] - A[5,9] * Conv_1_Relu_ts[9,14]
   + A[6,5] * Conv_2_aggregated_features[5,14]
   + A[7,5] * Conv_2_aggregated_features[5,14]
   + A[8,5] * Conv_2_aggregated_features[5,14]
   + A[9,5] * Conv_2_aggregated_features[5,14] ] = 0
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
   - A[5,8] * Conv_1_Relu_ts[8,15] - A[5,9] * Conv_1_Relu_ts[9,15]
   + A[6,5] * Conv_2_aggregated_features[5,15]
   + A[7,5] * Conv_2_aggregated_features[5,15]
   + A[8,5] * Conv_2_aggregated_features[5,15]
   + A[9,5] * Conv_2_aggregated_features[5,15] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,0] - A[6,8] * Conv_1_Relu_ts[8,0]
   - A[6,9] * Conv_1_Relu_ts[9,0]
   + A[7,6] * Conv_2_aggregated_features[6,0]
   + A[8,6] * Conv_2_aggregated_features[6,0]
   + A[9,6] * Conv_2_aggregated_features[6,0] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,1] - A[6,8] * Conv_1_Relu_ts[8,1]
   - A[6,9] * Conv_1_Relu_ts[9,1]
   + A[7,6] * Conv_2_aggregated_features[6,1]
   + A[8,6] * Conv_2_aggregated_features[6,1]
   + A[9,6] * Conv_2_aggregated_features[6,1] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,2] - A[6,8] * Conv_1_Relu_ts[8,2]
   - A[6,9] * Conv_1_Relu_ts[9,2]
   + A[7,6] * Conv_2_aggregated_features[6,2]
   + A[8,6] * Conv_2_aggregated_features[6,2]
   + A[9,6] * Conv_2_aggregated_features[6,2] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,3] - A[6,8] * Conv_1_Relu_ts[8,3]
   - A[6,9] * Conv_1_Relu_ts[9,3]
   + A[7,6] * Conv_2_aggregated_features[6,3]
   + A[8,6] * Conv_2_aggregated_features[6,3]
   + A[9,6] * Conv_2_aggregated_features[6,3] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,4] - A[6,8] * Conv_1_Relu_ts[8,4]
   - A[6,9] * Conv_1_Relu_ts[9,4]
   + A[7,6] * Conv_2_aggregated_features[6,4]
   + A[8,6] * Conv_2_aggregated_features[6,4]
   + A[9,6] * Conv_2_aggregated_features[6,4] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,5] - A[6,8] * Conv_1_Relu_ts[8,5]
   - A[6,9] * Conv_1_Relu_ts[9,5]
   + A[7,6] * Conv_2_aggregated_features[6,5]
   + A[8,6] * Conv_2_aggregated_features[6,5]
   + A[9,6] * Conv_2_aggregated_features[6,5] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,6] - A[6,8] * Conv_1_Relu_ts[8,6]
   - A[6,9] * Conv_1_Relu_ts[9,6]
   + A[7,6] * Conv_2_aggregated_features[6,6]
   + A[8,6] * Conv_2_aggregated_features[6,6]
   + A[9,6] * Conv_2_aggregated_features[6,6] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,7] - A[6,8] * Conv_1_Relu_ts[8,7]
   - A[6,9] * Conv_1_Relu_ts[9,7]
   + A[7,6] * Conv_2_aggregated_features[6,7]
   + A[8,6] * Conv_2_aggregated_features[6,7]
   + A[9,6] * Conv_2_aggregated_features[6,7] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,8] - A[6,8] * Conv_1_Relu_ts[8,8]
   - A[6,9] * Conv_1_Relu_ts[9,8]
   + A[7,6] * Conv_2_aggregated_features[6,8]
   + A[8,6] * Conv_2_aggregated_features[6,8]
   + A[9,6] * Conv_2_aggregated_features[6,8] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,9] - A[6,8] * Conv_1_Relu_ts[8,9]
   - A[6,9] * Conv_1_Relu_ts[9,9]
   + A[7,6] * Conv_2_aggregated_features[6,9]
   + A[8,6] * Conv_2_aggregated_features[6,9]
   + A[9,6] * Conv_2_aggregated_features[6,9] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,10] - A[6,8] * Conv_1_Relu_ts[8,10]
   - A[6,9] * Conv_1_Relu_ts[9,10]
   + A[7,6] * Conv_2_aggregated_features[6,10]
   + A[8,6] * Conv_2_aggregated_features[6,10]
   + A[9,6] * Conv_2_aggregated_features[6,10] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,11] - A[6,8] * Conv_1_Relu_ts[8,11]
   - A[6,9] * Conv_1_Relu_ts[9,11]
   + A[7,6] * Conv_2_aggregated_features[6,11]
   + A[8,6] * Conv_2_aggregated_features[6,11]
   + A[9,6] * Conv_2_aggregated_features[6,11] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,12] - A[6,8] * Conv_1_Relu_ts[8,12]
   - A[6,9] * Conv_1_Relu_ts[9,12]
   + A[7,6] * Conv_2_aggregated_features[6,12]
   + A[8,6] * Conv_2_aggregated_features[6,12]
   + A[9,6] * Conv_2_aggregated_features[6,12] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,13] - A[6,8] * Conv_1_Relu_ts[8,13]
   - A[6,9] * Conv_1_Relu_ts[9,13]
   + A[7,6] * Conv_2_aggregated_features[6,13]
   + A[8,6] * Conv_2_aggregated_features[6,13]
   + A[9,6] * Conv_2_aggregated_features[6,13] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,14] - A[6,8] * Conv_1_Relu_ts[8,14]
   - A[6,9] * Conv_1_Relu_ts[9,14]
   + A[7,6] * Conv_2_aggregated_features[6,14]
   + A[8,6] * Conv_2_aggregated_features[6,14]
   + A[9,6] * Conv_2_aggregated_features[6,14] ] = 0
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
   - A[6,7] * Conv_1_Relu_ts[7,15] - A[6,8] * Conv_1_Relu_ts[8,15]
   - A[6,9] * Conv_1_Relu_ts[9,15]
   + A[7,6] * Conv_2_aggregated_features[6,15]
   + A[8,6] * Conv_2_aggregated_features[6,15]
   + A[9,6] * Conv_2_aggregated_features[6,15] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,0]
   - A[7,8] * Conv_1_Relu_ts[8,0] - A[7,9] * Conv_1_Relu_ts[9,0]
   + A[8,7] * Conv_2_aggregated_features[7,0]
   + A[9,7] * Conv_2_aggregated_features[7,0] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,1]
   - A[7,8] * Conv_1_Relu_ts[8,1] - A[7,9] * Conv_1_Relu_ts[9,1]
   + A[8,7] * Conv_2_aggregated_features[7,1]
   + A[9,7] * Conv_2_aggregated_features[7,1] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,2]
   - A[7,8] * Conv_1_Relu_ts[8,2] - A[7,9] * Conv_1_Relu_ts[9,2]
   + A[8,7] * Conv_2_aggregated_features[7,2]
   + A[9,7] * Conv_2_aggregated_features[7,2] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,3]
   - A[7,8] * Conv_1_Relu_ts[8,3] - A[7,9] * Conv_1_Relu_ts[9,3]
   + A[8,7] * Conv_2_aggregated_features[7,3]
   + A[9,7] * Conv_2_aggregated_features[7,3] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,4]
   - A[7,8] * Conv_1_Relu_ts[8,4] - A[7,9] * Conv_1_Relu_ts[9,4]
   + A[8,7] * Conv_2_aggregated_features[7,4]
   + A[9,7] * Conv_2_aggregated_features[7,4] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,5]
   - A[7,8] * Conv_1_Relu_ts[8,5] - A[7,9] * Conv_1_Relu_ts[9,5]
   + A[8,7] * Conv_2_aggregated_features[7,5]
   + A[9,7] * Conv_2_aggregated_features[7,5] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,6]
   - A[7,8] * Conv_1_Relu_ts[8,6] - A[7,9] * Conv_1_Relu_ts[9,6]
   + A[8,7] * Conv_2_aggregated_features[7,6]
   + A[9,7] * Conv_2_aggregated_features[7,6] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,7]
   - A[7,8] * Conv_1_Relu_ts[8,7] - A[7,9] * Conv_1_Relu_ts[9,7]
   + A[8,7] * Conv_2_aggregated_features[7,7]
   + A[9,7] * Conv_2_aggregated_features[7,7] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,8]
   - A[7,8] * Conv_1_Relu_ts[8,8] - A[7,9] * Conv_1_Relu_ts[9,8]
   + A[8,7] * Conv_2_aggregated_features[7,8]
   + A[9,7] * Conv_2_aggregated_features[7,8] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,9]
   - A[7,8] * Conv_1_Relu_ts[8,9] - A[7,9] * Conv_1_Relu_ts[9,9]
   + A[8,7] * Conv_2_aggregated_features[7,9]
   + A[9,7] * Conv_2_aggregated_features[7,9] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,10]
   - A[7,8] * Conv_1_Relu_ts[8,10] - A[7,9] * Conv_1_Relu_ts[9,10]
   + A[8,7] * Conv_2_aggregated_features[7,10]
   + A[9,7] * Conv_2_aggregated_features[7,10] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,11]
   - A[7,8] * Conv_1_Relu_ts[8,11] - A[7,9] * Conv_1_Relu_ts[9,11]
   + A[8,7] * Conv_2_aggregated_features[7,11]
   + A[9,7] * Conv_2_aggregated_features[7,11] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,12]
   - A[7,8] * Conv_1_Relu_ts[8,12] - A[7,9] * Conv_1_Relu_ts[9,12]
   + A[8,7] * Conv_2_aggregated_features[7,12]
   + A[9,7] * Conv_2_aggregated_features[7,12] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,13]
   - A[7,8] * Conv_1_Relu_ts[8,13] - A[7,9] * Conv_1_Relu_ts[9,13]
   + A[8,7] * Conv_2_aggregated_features[7,13]
   + A[9,7] * Conv_2_aggregated_features[7,13] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,14]
   - A[7,8] * Conv_1_Relu_ts[8,14] - A[7,9] * Conv_1_Relu_ts[9,14]
   + A[8,7] * Conv_2_aggregated_features[7,14]
   + A[9,7] * Conv_2_aggregated_features[7,14] ] = 0
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
   + A[7,7] * Conv_2_aggregated_features[7,15]
   - A[7,8] * Conv_1_Relu_ts[8,15] - A[7,9] * Conv_1_Relu_ts[9,15]
   + A[8,7] * Conv_2_aggregated_features[7,15]
   + A[9,7] * Conv_2_aggregated_features[7,15] ] = 0
 Conv_2_averages_constraint[8,0]: [
   A[0,8] * Conv_2_aggregated_features[8,0]
   + A[1,8] * Conv_2_aggregated_features[8,0]
   + A[2,8] * Conv_2_aggregated_features[8,0]
   + A[3,8] * Conv_2_aggregated_features[8,0]
   + A[4,8] * Conv_2_aggregated_features[8,0]
   + A[5,8] * Conv_2_aggregated_features[8,0]
   + A[6,8] * Conv_2_aggregated_features[8,0]
   + A[7,8] * Conv_2_aggregated_features[8,0]
   - A[8,0] * Conv_1_Relu_ts[0,0] - A[8,1] * Conv_1_Relu_ts[1,0]
   - A[8,2] * Conv_1_Relu_ts[2,0] - A[8,3] * Conv_1_Relu_ts[3,0]
   - A[8,4] * Conv_1_Relu_ts[4,0] - A[8,5] * Conv_1_Relu_ts[5,0]
   - A[8,6] * Conv_1_Relu_ts[6,0] - A[8,7] * Conv_1_Relu_ts[7,0]
   - A[8,8] * Conv_1_Relu_ts[8,0]
   + A[8,8] * Conv_2_aggregated_features[8,0]
   - A[8,9] * Conv_1_Relu_ts[9,0]
   + A[9,8] * Conv_2_aggregated_features[8,0] ] = 0
 Conv_2_averages_constraint[8,1]: [
   A[0,8] * Conv_2_aggregated_features[8,1]
   + A[1,8] * Conv_2_aggregated_features[8,1]
   + A[2,8] * Conv_2_aggregated_features[8,1]
   + A[3,8] * Conv_2_aggregated_features[8,1]
   + A[4,8] * Conv_2_aggregated_features[8,1]
   + A[5,8] * Conv_2_aggregated_features[8,1]
   + A[6,8] * Conv_2_aggregated_features[8,1]
   + A[7,8] * Conv_2_aggregated_features[8,1]
   - A[8,0] * Conv_1_Relu_ts[0,1] - A[8,1] * Conv_1_Relu_ts[1,1]
   - A[8,2] * Conv_1_Relu_ts[2,1] - A[8,3] * Conv_1_Relu_ts[3,1]
   - A[8,4] * Conv_1_Relu_ts[4,1] - A[8,5] * Conv_1_Relu_ts[5,1]
   - A[8,6] * Conv_1_Relu_ts[6,1] - A[8,7] * Conv_1_Relu_ts[7,1]
   - A[8,8] * Conv_1_Relu_ts[8,1]
   + A[8,8] * Conv_2_aggregated_features[8,1]
   - A[8,9] * Conv_1_Relu_ts[9,1]
   + A[9,8] * Conv_2_aggregated_features[8,1] ] = 0
 Conv_2_averages_constraint[8,2]: [
   A[0,8] * Conv_2_aggregated_features[8,2]
   + A[1,8] * Conv_2_aggregated_features[8,2]
   + A[2,8] * Conv_2_aggregated_features[8,2]
   + A[3,8] * Conv_2_aggregated_features[8,2]
   + A[4,8] * Conv_2_aggregated_features[8,2]
   + A[5,8] * Conv_2_aggregated_features[8,2]
   + A[6,8] * Conv_2_aggregated_features[8,2]
   + A[7,8] * Conv_2_aggregated_features[8,2]
   - A[8,0] * Conv_1_Relu_ts[0,2] - A[8,1] * Conv_1_Relu_ts[1,2]
   - A[8,2] * Conv_1_Relu_ts[2,2] - A[8,3] * Conv_1_Relu_ts[3,2]
   - A[8,4] * Conv_1_Relu_ts[4,2] - A[8,5] * Conv_1_Relu_ts[5,2]
   - A[8,6] * Conv_1_Relu_ts[6,2] - A[8,7] * Conv_1_Relu_ts[7,2]
   - A[8,8] * Conv_1_Relu_ts[8,2]
   + A[8,8] * Conv_2_aggregated_features[8,2]
   - A[8,9] * Conv_1_Relu_ts[9,2]
   + A[9,8] * Conv_2_aggregated_features[8,2] ] = 0
 Conv_2_averages_constraint[8,3]: [
   A[0,8] * Conv_2_aggregated_features[8,3]
   + A[1,8] * Conv_2_aggregated_features[8,3]
   + A[2,8] * Conv_2_aggregated_features[8,3]
   + A[3,8] * Conv_2_aggregated_features[8,3]
   + A[4,8] * Conv_2_aggregated_features[8,3]
   + A[5,8] * Conv_2_aggregated_features[8,3]
   + A[6,8] * Conv_2_aggregated_features[8,3]
   + A[7,8] * Conv_2_aggregated_features[8,3]
   - A[8,0] * Conv_1_Relu_ts[0,3] - A[8,1] * Conv_1_Relu_ts[1,3]
   - A[8,2] * Conv_1_Relu_ts[2,3] - A[8,3] * Conv_1_Relu_ts[3,3]
   - A[8,4] * Conv_1_Relu_ts[4,3] - A[8,5] * Conv_1_Relu_ts[5,3]
   - A[8,6] * Conv_1_Relu_ts[6,3] - A[8,7] * Conv_1_Relu_ts[7,3]
   - A[8,8] * Conv_1_Relu_ts[8,3]
   + A[8,8] * Conv_2_aggregated_features[8,3]
   - A[8,9] * Conv_1_Relu_ts[9,3]
   + A[9,8] * Conv_2_aggregated_features[8,3] ] = 0
 Conv_2_averages_constraint[8,4]: [
   A[0,8] * Conv_2_aggregated_features[8,4]
   + A[1,8] * Conv_2_aggregated_features[8,4]
   + A[2,8] * Conv_2_aggregated_features[8,4]
   + A[3,8] * Conv_2_aggregated_features[8,4]
   + A[4,8] * Conv_2_aggregated_features[8,4]
   + A[5,8] * Conv_2_aggregated_features[8,4]
   + A[6,8] * Conv_2_aggregated_features[8,4]
   + A[7,8] * Conv_2_aggregated_features[8,4]
   - A[8,0] * Conv_1_Relu_ts[0,4] - A[8,1] * Conv_1_Relu_ts[1,4]
   - A[8,2] * Conv_1_Relu_ts[2,4] - A[8,3] * Conv_1_Relu_ts[3,4]
   - A[8,4] * Conv_1_Relu_ts[4,4] - A[8,5] * Conv_1_Relu_ts[5,4]
   - A[8,6] * Conv_1_Relu_ts[6,4] - A[8,7] * Conv_1_Relu_ts[7,4]
   - A[8,8] * Conv_1_Relu_ts[8,4]
   + A[8,8] * Conv_2_aggregated_features[8,4]
   - A[8,9] * Conv_1_Relu_ts[9,4]
   + A[9,8] * Conv_2_aggregated_features[8,4] ] = 0
 Conv_2_averages_constraint[8,5]: [
   A[0,8] * Conv_2_aggregated_features[8,5]
   + A[1,8] * Conv_2_aggregated_features[8,5]
   + A[2,8] * Conv_2_aggregated_features[8,5]
   + A[3,8] * Conv_2_aggregated_features[8,5]
   + A[4,8] * Conv_2_aggregated_features[8,5]
   + A[5,8] * Conv_2_aggregated_features[8,5]
   + A[6,8] * Conv_2_aggregated_features[8,5]
   + A[7,8] * Conv_2_aggregated_features[8,5]
   - A[8,0] * Conv_1_Relu_ts[0,5] - A[8,1] * Conv_1_Relu_ts[1,5]
   - A[8,2] * Conv_1_Relu_ts[2,5] - A[8,3] * Conv_1_Relu_ts[3,5]
   - A[8,4] * Conv_1_Relu_ts[4,5] - A[8,5] * Conv_1_Relu_ts[5,5]
   - A[8,6] * Conv_1_Relu_ts[6,5] - A[8,7] * Conv_1_Relu_ts[7,5]
   - A[8,8] * Conv_1_Relu_ts[8,5]
   + A[8,8] * Conv_2_aggregated_features[8,5]
   - A[8,9] * Conv_1_Relu_ts[9,5]
   + A[9,8] * Conv_2_aggregated_features[8,5] ] = 0
 Conv_2_averages_constraint[8,6]: [
   A[0,8] * Conv_2_aggregated_features[8,6]
   + A[1,8] * Conv_2_aggregated_features[8,6]
   + A[2,8] * Conv_2_aggregated_features[8,6]
   + A[3,8] * Conv_2_aggregated_features[8,6]
   + A[4,8] * Conv_2_aggregated_features[8,6]
   + A[5,8] * Conv_2_aggregated_features[8,6]
   + A[6,8] * Conv_2_aggregated_features[8,6]
   + A[7,8] * Conv_2_aggregated_features[8,6]
   - A[8,0] * Conv_1_Relu_ts[0,6] - A[8,1] * Conv_1_Relu_ts[1,6]
   - A[8,2] * Conv_1_Relu_ts[2,6] - A[8,3] * Conv_1_Relu_ts[3,6]
   - A[8,4] * Conv_1_Relu_ts[4,6] - A[8,5] * Conv_1_Relu_ts[5,6]
   - A[8,6] * Conv_1_Relu_ts[6,6] - A[8,7] * Conv_1_Relu_ts[7,6]
   - A[8,8] * Conv_1_Relu_ts[8,6]
   + A[8,8] * Conv_2_aggregated_features[8,6]
   - A[8,9] * Conv_1_Relu_ts[9,6]
   + A[9,8] * Conv_2_aggregated_features[8,6] ] = 0
 Conv_2_averages_constraint[8,7]: [
   A[0,8] * Conv_2_aggregated_features[8,7]
   + A[1,8] * Conv_2_aggregated_features[8,7]
   + A[2,8] * Conv_2_aggregated_features[8,7]
   + A[3,8] * Conv_2_aggregated_features[8,7]
   + A[4,8] * Conv_2_aggregated_features[8,7]
   + A[5,8] * Conv_2_aggregated_features[8,7]
   + A[6,8] * Conv_2_aggregated_features[8,7]
   + A[7,8] * Conv_2_aggregated_features[8,7]
   - A[8,0] * Conv_1_Relu_ts[0,7] - A[8,1] * Conv_1_Relu_ts[1,7]
   - A[8,2] * Conv_1_Relu_ts[2,7] - A[8,3] * Conv_1_Relu_ts[3,7]
   - A[8,4] * Conv_1_Relu_ts[4,7] - A[8,5] * Conv_1_Relu_ts[5,7]
   - A[8,6] * Conv_1_Relu_ts[6,7] - A[8,7] * Conv_1_Relu_ts[7,7]
   - A[8,8] * Conv_1_Relu_ts[8,7]
   + A[8,8] * Conv_2_aggregated_features[8,7]
   - A[8,9] * Conv_1_Relu_ts[9,7]
   + A[9,8] * Conv_2_aggregated_features[8,7] ] = 0
 Conv_2_averages_constraint[8,8]: [
   A[0,8] * Conv_2_aggregated_features[8,8]
   + A[1,8] * Conv_2_aggregated_features[8,8]
   + A[2,8] * Conv_2_aggregated_features[8,8]
   + A[3,8] * Conv_2_aggregated_features[8,8]
   + A[4,8] * Conv_2_aggregated_features[8,8]
   + A[5,8] * Conv_2_aggregated_features[8,8]
   + A[6,8] * Conv_2_aggregated_features[8,8]
   + A[7,8] * Conv_2_aggregated_features[8,8]
   - A[8,0] * Conv_1_Relu_ts[0,8] - A[8,1] * Conv_1_Relu_ts[1,8]
   - A[8,2] * Conv_1_Relu_ts[2,8] - A[8,3] * Conv_1_Relu_ts[3,8]
   - A[8,4] * Conv_1_Relu_ts[4,8] - A[8,5] * Conv_1_Relu_ts[5,8]
   - A[8,6] * Conv_1_Relu_ts[6,8] - A[8,7] * Conv_1_Relu_ts[7,8]
   - A[8,8] * Conv_1_Relu_ts[8,8]
   + A[8,8] * Conv_2_aggregated_features[8,8]
   - A[8,9] * Conv_1_Relu_ts[9,8]
   + A[9,8] * Conv_2_aggregated_features[8,8] ] = 0
 Conv_2_averages_constraint[8,9]: [
   A[0,8] * Conv_2_aggregated_features[8,9]
   + A[1,8] * Conv_2_aggregated_features[8,9]
   + A[2,8] * Conv_2_aggregated_features[8,9]
   + A[3,8] * Conv_2_aggregated_features[8,9]
   + A[4,8] * Conv_2_aggregated_features[8,9]
   + A[5,8] * Conv_2_aggregated_features[8,9]
   + A[6,8] * Conv_2_aggregated_features[8,9]
   + A[7,8] * Conv_2_aggregated_features[8,9]
   - A[8,0] * Conv_1_Relu_ts[0,9] - A[8,1] * Conv_1_Relu_ts[1,9]
   - A[8,2] * Conv_1_Relu_ts[2,9] - A[8,3] * Conv_1_Relu_ts[3,9]
   - A[8,4] * Conv_1_Relu_ts[4,9] - A[8,5] * Conv_1_Relu_ts[5,9]
   - A[8,6] * Conv_1_Relu_ts[6,9] - A[8,7] * Conv_1_Relu_ts[7,9]
   - A[8,8] * Conv_1_Relu_ts[8,9]
   + A[8,8] * Conv_2_aggregated_features[8,9]
   - A[8,9] * Conv_1_Relu_ts[9,9]
   + A[9,8] * Conv_2_aggregated_features[8,9] ] = 0
 Conv_2_averages_constraint[8,10]: [
   A[0,8] * Conv_2_aggregated_features[8,10]
   + A[1,8] * Conv_2_aggregated_features[8,10]
   + A[2,8] * Conv_2_aggregated_features[8,10]
   + A[3,8] * Conv_2_aggregated_features[8,10]
   + A[4,8] * Conv_2_aggregated_features[8,10]
   + A[5,8] * Conv_2_aggregated_features[8,10]
   + A[6,8] * Conv_2_aggregated_features[8,10]
   + A[7,8] * Conv_2_aggregated_features[8,10]
   - A[8,0] * Conv_1_Relu_ts[0,10] - A[8,1] * Conv_1_Relu_ts[1,10]
   - A[8,2] * Conv_1_Relu_ts[2,10] - A[8,3] * Conv_1_Relu_ts[3,10]
   - A[8,4] * Conv_1_Relu_ts[4,10] - A[8,5] * Conv_1_Relu_ts[5,10]
   - A[8,6] * Conv_1_Relu_ts[6,10] - A[8,7] * Conv_1_Relu_ts[7,10]
   - A[8,8] * Conv_1_Relu_ts[8,10]
   + A[8,8] * Conv_2_aggregated_features[8,10]
   - A[8,9] * Conv_1_Relu_ts[9,10]
   + A[9,8] * Conv_2_aggregated_features[8,10] ] = 0
 Conv_2_averages_constraint[8,11]: [
   A[0,8] * Conv_2_aggregated_features[8,11]
   + A[1,8] * Conv_2_aggregated_features[8,11]
   + A[2,8] * Conv_2_aggregated_features[8,11]
   + A[3,8] * Conv_2_aggregated_features[8,11]
   + A[4,8] * Conv_2_aggregated_features[8,11]
   + A[5,8] * Conv_2_aggregated_features[8,11]
   + A[6,8] * Conv_2_aggregated_features[8,11]
   + A[7,8] * Conv_2_aggregated_features[8,11]
   - A[8,0] * Conv_1_Relu_ts[0,11] - A[8,1] * Conv_1_Relu_ts[1,11]
   - A[8,2] * Conv_1_Relu_ts[2,11] - A[8,3] * Conv_1_Relu_ts[3,11]
   - A[8,4] * Conv_1_Relu_ts[4,11] - A[8,5] * Conv_1_Relu_ts[5,11]
   - A[8,6] * Conv_1_Relu_ts[6,11] - A[8,7] * Conv_1_Relu_ts[7,11]
   - A[8,8] * Conv_1_Relu_ts[8,11]
   + A[8,8] * Conv_2_aggregated_features[8,11]
   - A[8,9] * Conv_1_Relu_ts[9,11]
   + A[9,8] * Conv_2_aggregated_features[8,11] ] = 0
 Conv_2_averages_constraint[8,12]: [
   A[0,8] * Conv_2_aggregated_features[8,12]
   + A[1,8] * Conv_2_aggregated_features[8,12]
   + A[2,8] * Conv_2_aggregated_features[8,12]
   + A[3,8] * Conv_2_aggregated_features[8,12]
   + A[4,8] * Conv_2_aggregated_features[8,12]
   + A[5,8] * Conv_2_aggregated_features[8,12]
   + A[6,8] * Conv_2_aggregated_features[8,12]
   + A[7,8] * Conv_2_aggregated_features[8,12]
   - A[8,0] * Conv_1_Relu_ts[0,12] - A[8,1] * Conv_1_Relu_ts[1,12]
   - A[8,2] * Conv_1_Relu_ts[2,12] - A[8,3] * Conv_1_Relu_ts[3,12]
   - A[8,4] * Conv_1_Relu_ts[4,12] - A[8,5] * Conv_1_Relu_ts[5,12]
   - A[8,6] * Conv_1_Relu_ts[6,12] - A[8,7] * Conv_1_Relu_ts[7,12]
   - A[8,8] * Conv_1_Relu_ts[8,12]
   + A[8,8] * Conv_2_aggregated_features[8,12]
   - A[8,9] * Conv_1_Relu_ts[9,12]
   + A[9,8] * Conv_2_aggregated_features[8,12] ] = 0
 Conv_2_averages_constraint[8,13]: [
   A[0,8] * Conv_2_aggregated_features[8,13]
   + A[1,8] * Conv_2_aggregated_features[8,13]
   + A[2,8] * Conv_2_aggregated_features[8,13]
   + A[3,8] * Conv_2_aggregated_features[8,13]
   + A[4,8] * Conv_2_aggregated_features[8,13]
   + A[5,8] * Conv_2_aggregated_features[8,13]
   + A[6,8] * Conv_2_aggregated_features[8,13]
   + A[7,8] * Conv_2_aggregated_features[8,13]
   - A[8,0] * Conv_1_Relu_ts[0,13] - A[8,1] * Conv_1_Relu_ts[1,13]
   - A[8,2] * Conv_1_Relu_ts[2,13] - A[8,3] * Conv_1_Relu_ts[3,13]
   - A[8,4] * Conv_1_Relu_ts[4,13] - A[8,5] * Conv_1_Relu_ts[5,13]
   - A[8,6] * Conv_1_Relu_ts[6,13] - A[8,7] * Conv_1_Relu_ts[7,13]
   - A[8,8] * Conv_1_Relu_ts[8,13]
   + A[8,8] * Conv_2_aggregated_features[8,13]
   - A[8,9] * Conv_1_Relu_ts[9,13]
   + A[9,8] * Conv_2_aggregated_features[8,13] ] = 0
 Conv_2_averages_constraint[8,14]: [
   A[0,8] * Conv_2_aggregated_features[8,14]
   + A[1,8] * Conv_2_aggregated_features[8,14]
   + A[2,8] * Conv_2_aggregated_features[8,14]
   + A[3,8] * Conv_2_aggregated_features[8,14]
   + A[4,8] * Conv_2_aggregated_features[8,14]
   + A[5,8] * Conv_2_aggregated_features[8,14]
   + A[6,8] * Conv_2_aggregated_features[8,14]
   + A[7,8] * Conv_2_aggregated_features[8,14]
   - A[8,0] * Conv_1_Relu_ts[0,14] - A[8,1] * Conv_1_Relu_ts[1,14]
   - A[8,2] * Conv_1_Relu_ts[2,14] - A[8,3] * Conv_1_Relu_ts[3,14]
   - A[8,4] * Conv_1_Relu_ts[4,14] - A[8,5] * Conv_1_Relu_ts[5,14]
   - A[8,6] * Conv_1_Relu_ts[6,14] - A[8,7] * Conv_1_Relu_ts[7,14]
   - A[8,8] * Conv_1_Relu_ts[8,14]
   + A[8,8] * Conv_2_aggregated_features[8,14]
   - A[8,9] * Conv_1_Relu_ts[9,14]
   + A[9,8] * Conv_2_aggregated_features[8,14] ] = 0
 Conv_2_averages_constraint[8,15]: [
   A[0,8] * Conv_2_aggregated_features[8,15]
   + A[1,8] * Conv_2_aggregated_features[8,15]
   + A[2,8] * Conv_2_aggregated_features[8,15]
   + A[3,8] * Conv_2_aggregated_features[8,15]
   + A[4,8] * Conv_2_aggregated_features[8,15]
   + A[5,8] * Conv_2_aggregated_features[8,15]
   + A[6,8] * Conv_2_aggregated_features[8,15]
   + A[7,8] * Conv_2_aggregated_features[8,15]
   - A[8,0] * Conv_1_Relu_ts[0,15] - A[8,1] * Conv_1_Relu_ts[1,15]
   - A[8,2] * Conv_1_Relu_ts[2,15] - A[8,3] * Conv_1_Relu_ts[3,15]
   - A[8,4] * Conv_1_Relu_ts[4,15] - A[8,5] * Conv_1_Relu_ts[5,15]
   - A[8,6] * Conv_1_Relu_ts[6,15] - A[8,7] * Conv_1_Relu_ts[7,15]
   - A[8,8] * Conv_1_Relu_ts[8,15]
   + A[8,8] * Conv_2_aggregated_features[8,15]
   - A[8,9] * Conv_1_Relu_ts[9,15]
   + A[9,8] * Conv_2_aggregated_features[8,15] ] = 0
 Conv_2_averages_constraint[9,0]: [
   A[0,9] * Conv_2_aggregated_features[9,0]
   + A[1,9] * Conv_2_aggregated_features[9,0]
   + A[2,9] * Conv_2_aggregated_features[9,0]
   + A[3,9] * Conv_2_aggregated_features[9,0]
   + A[4,9] * Conv_2_aggregated_features[9,0]
   + A[5,9] * Conv_2_aggregated_features[9,0]
   + A[6,9] * Conv_2_aggregated_features[9,0]
   + A[7,9] * Conv_2_aggregated_features[9,0]
   + A[8,9] * Conv_2_aggregated_features[9,0]
   - A[9,0] * Conv_1_Relu_ts[0,0] - A[9,1] * Conv_1_Relu_ts[1,0]
   - A[9,2] * Conv_1_Relu_ts[2,0] - A[9,3] * Conv_1_Relu_ts[3,0]
   - A[9,4] * Conv_1_Relu_ts[4,0] - A[9,5] * Conv_1_Relu_ts[5,0]
   - A[9,6] * Conv_1_Relu_ts[6,0] - A[9,7] * Conv_1_Relu_ts[7,0]
   - A[9,8] * Conv_1_Relu_ts[8,0] - A[9,9] * Conv_1_Relu_ts[9,0]
   + A[9,9] * Conv_2_aggregated_features[9,0] ] = 0
 Conv_2_averages_constraint[9,1]: [
   A[0,9] * Conv_2_aggregated_features[9,1]
   + A[1,9] * Conv_2_aggregated_features[9,1]
   + A[2,9] * Conv_2_aggregated_features[9,1]
   + A[3,9] * Conv_2_aggregated_features[9,1]
   + A[4,9] * Conv_2_aggregated_features[9,1]
   + A[5,9] * Conv_2_aggregated_features[9,1]
   + A[6,9] * Conv_2_aggregated_features[9,1]
   + A[7,9] * Conv_2_aggregated_features[9,1]
   + A[8,9] * Conv_2_aggregated_features[9,1]
   - A[9,0] * Conv_1_Relu_ts[0,1] - A[9,1] * Conv_1_Relu_ts[1,1]
   - A[9,2] * Conv_1_Relu_ts[2,1] - A[9,3] * Conv_1_Relu_ts[3,1]
   - A[9,4] * Conv_1_Relu_ts[4,1] - A[9,5] * Conv_1_Relu_ts[5,1]
   - A[9,6] * Conv_1_Relu_ts[6,1] - A[9,7] * Conv_1_Relu_ts[7,1]
   - A[9,8] * Conv_1_Relu_ts[8,1] - A[9,9] * Conv_1_Relu_ts[9,1]
   + A[9,9] * Conv_2_aggregated_features[9,1] ] = 0
 Conv_2_averages_constraint[9,2]: [
   A[0,9] * Conv_2_aggregated_features[9,2]
   + A[1,9] * Conv_2_aggregated_features[9,2]
   + A[2,9] * Conv_2_aggregated_features[9,2]
   + A[3,9] * Conv_2_aggregated_features[9,2]
   + A[4,9] * Conv_2_aggregated_features[9,2]
   + A[5,9] * Conv_2_aggregated_features[9,2]
   + A[6,9] * Conv_2_aggregated_features[9,2]
   + A[7,9] * Conv_2_aggregated_features[9,2]
   + A[8,9] * Conv_2_aggregated_features[9,2]
   - A[9,0] * Conv_1_Relu_ts[0,2] - A[9,1] * Conv_1_Relu_ts[1,2]
   - A[9,2] * Conv_1_Relu_ts[2,2] - A[9,3] * Conv_1_Relu_ts[3,2]
   - A[9,4] * Conv_1_Relu_ts[4,2] - A[9,5] * Conv_1_Relu_ts[5,2]
   - A[9,6] * Conv_1_Relu_ts[6,2] - A[9,7] * Conv_1_Relu_ts[7,2]
   - A[9,8] * Conv_1_Relu_ts[8,2] - A[9,9] * Conv_1_Relu_ts[9,2]
   + A[9,9] * Conv_2_aggregated_features[9,2] ] = 0
 Conv_2_averages_constraint[9,3]: [
   A[0,9] * Conv_2_aggregated_features[9,3]
   + A[1,9] * Conv_2_aggregated_features[9,3]
   + A[2,9] * Conv_2_aggregated_features[9,3]
   + A[3,9] * Conv_2_aggregated_features[9,3]
   + A[4,9] * Conv_2_aggregated_features[9,3]
   + A[5,9] * Conv_2_aggregated_features[9,3]
   + A[6,9] * Conv_2_aggregated_features[9,3]
   + A[7,9] * Conv_2_aggregated_features[9,3]
   + A[8,9] * Conv_2_aggregated_features[9,3]
   - A[9,0] * Conv_1_Relu_ts[0,3] - A[9,1] * Conv_1_Relu_ts[1,3]
   - A[9,2] * Conv_1_Relu_ts[2,3] - A[9,3] * Conv_1_Relu_ts[3,3]
   - A[9,4] * Conv_1_Relu_ts[4,3] - A[9,5] * Conv_1_Relu_ts[5,3]
   - A[9,6] * Conv_1_Relu_ts[6,3] - A[9,7] * Conv_1_Relu_ts[7,3]
   - A[9,8] * Conv_1_Relu_ts[8,3] - A[9,9] * Conv_1_Relu_ts[9,3]
   + A[9,9] * Conv_2_aggregated_features[9,3] ] = 0
 Conv_2_averages_constraint[9,4]: [
   A[0,9] * Conv_2_aggregated_features[9,4]
   + A[1,9] * Conv_2_aggregated_features[9,4]
   + A[2,9] * Conv_2_aggregated_features[9,4]
   + A[3,9] * Conv_2_aggregated_features[9,4]
   + A[4,9] * Conv_2_aggregated_features[9,4]
   + A[5,9] * Conv_2_aggregated_features[9,4]
   + A[6,9] * Conv_2_aggregated_features[9,4]
   + A[7,9] * Conv_2_aggregated_features[9,4]
   + A[8,9] * Conv_2_aggregated_features[9,4]
   - A[9,0] * Conv_1_Relu_ts[0,4] - A[9,1] * Conv_1_Relu_ts[1,4]
   - A[9,2] * Conv_1_Relu_ts[2,4] - A[9,3] * Conv_1_Relu_ts[3,4]
   - A[9,4] * Conv_1_Relu_ts[4,4] - A[9,5] * Conv_1_Relu_ts[5,4]
   - A[9,6] * Conv_1_Relu_ts[6,4] - A[9,7] * Conv_1_Relu_ts[7,4]
   - A[9,8] * Conv_1_Relu_ts[8,4] - A[9,9] * Conv_1_Relu_ts[9,4]
   + A[9,9] * Conv_2_aggregated_features[9,4] ] = 0
 Conv_2_averages_constraint[9,5]: [
   A[0,9] * Conv_2_aggregated_features[9,5]
   + A[1,9] * Conv_2_aggregated_features[9,5]
   + A[2,9] * Conv_2_aggregated_features[9,5]
   + A[3,9] * Conv_2_aggregated_features[9,5]
   + A[4,9] * Conv_2_aggregated_features[9,5]
   + A[5,9] * Conv_2_aggregated_features[9,5]
   + A[6,9] * Conv_2_aggregated_features[9,5]
   + A[7,9] * Conv_2_aggregated_features[9,5]
   + A[8,9] * Conv_2_aggregated_features[9,5]
   - A[9,0] * Conv_1_Relu_ts[0,5] - A[9,1] * Conv_1_Relu_ts[1,5]
   - A[9,2] * Conv_1_Relu_ts[2,5] - A[9,3] * Conv_1_Relu_ts[3,5]
   - A[9,4] * Conv_1_Relu_ts[4,5] - A[9,5] * Conv_1_Relu_ts[5,5]
   - A[9,6] * Conv_1_Relu_ts[6,5] - A[9,7] * Conv_1_Relu_ts[7,5]
   - A[9,8] * Conv_1_Relu_ts[8,5] - A[9,9] * Conv_1_Relu_ts[9,5]
   + A[9,9] * Conv_2_aggregated_features[9,5] ] = 0
 Conv_2_averages_constraint[9,6]: [
   A[0,9] * Conv_2_aggregated_features[9,6]
   + A[1,9] * Conv_2_aggregated_features[9,6]
   + A[2,9] * Conv_2_aggregated_features[9,6]
   + A[3,9] * Conv_2_aggregated_features[9,6]
   + A[4,9] * Conv_2_aggregated_features[9,6]
   + A[5,9] * Conv_2_aggregated_features[9,6]
   + A[6,9] * Conv_2_aggregated_features[9,6]
   + A[7,9] * Conv_2_aggregated_features[9,6]
   + A[8,9] * Conv_2_aggregated_features[9,6]
   - A[9,0] * Conv_1_Relu_ts[0,6] - A[9,1] * Conv_1_Relu_ts[1,6]
   - A[9,2] * Conv_1_Relu_ts[2,6] - A[9,3] * Conv_1_Relu_ts[3,6]
   - A[9,4] * Conv_1_Relu_ts[4,6] - A[9,5] * Conv_1_Relu_ts[5,6]
   - A[9,6] * Conv_1_Relu_ts[6,6] - A[9,7] * Conv_1_Relu_ts[7,6]
   - A[9,8] * Conv_1_Relu_ts[8,6] - A[9,9] * Conv_1_Relu_ts[9,6]
   + A[9,9] * Conv_2_aggregated_features[9,6] ] = 0
 Conv_2_averages_constraint[9,7]: [
   A[0,9] * Conv_2_aggregated_features[9,7]
   + A[1,9] * Conv_2_aggregated_features[9,7]
   + A[2,9] * Conv_2_aggregated_features[9,7]
   + A[3,9] * Conv_2_aggregated_features[9,7]
   + A[4,9] * Conv_2_aggregated_features[9,7]
   + A[5,9] * Conv_2_aggregated_features[9,7]
   + A[6,9] * Conv_2_aggregated_features[9,7]
   + A[7,9] * Conv_2_aggregated_features[9,7]
   + A[8,9] * Conv_2_aggregated_features[9,7]
   - A[9,0] * Conv_1_Relu_ts[0,7] - A[9,1] * Conv_1_Relu_ts[1,7]
   - A[9,2] * Conv_1_Relu_ts[2,7] - A[9,3] * Conv_1_Relu_ts[3,7]
   - A[9,4] * Conv_1_Relu_ts[4,7] - A[9,5] * Conv_1_Relu_ts[5,7]
   - A[9,6] * Conv_1_Relu_ts[6,7] - A[9,7] * Conv_1_Relu_ts[7,7]
   - A[9,8] * Conv_1_Relu_ts[8,7] - A[9,9] * Conv_1_Relu_ts[9,7]
   + A[9,9] * Conv_2_aggregated_features[9,7] ] = 0
 Conv_2_averages_constraint[9,8]: [
   A[0,9] * Conv_2_aggregated_features[9,8]
   + A[1,9] * Conv_2_aggregated_features[9,8]
   + A[2,9] * Conv_2_aggregated_features[9,8]
   + A[3,9] * Conv_2_aggregated_features[9,8]
   + A[4,9] * Conv_2_aggregated_features[9,8]
   + A[5,9] * Conv_2_aggregated_features[9,8]
   + A[6,9] * Conv_2_aggregated_features[9,8]
   + A[7,9] * Conv_2_aggregated_features[9,8]
   + A[8,9] * Conv_2_aggregated_features[9,8]
   - A[9,0] * Conv_1_Relu_ts[0,8] - A[9,1] * Conv_1_Relu_ts[1,8]
   - A[9,2] * Conv_1_Relu_ts[2,8] - A[9,3] * Conv_1_Relu_ts[3,8]
   - A[9,4] * Conv_1_Relu_ts[4,8] - A[9,5] * Conv_1_Relu_ts[5,8]
   - A[9,6] * Conv_1_Relu_ts[6,8] - A[9,7] * Conv_1_Relu_ts[7,8]
   - A[9,8] * Conv_1_Relu_ts[8,8] - A[9,9] * Conv_1_Relu_ts[9,8]
   + A[9,9] * Conv_2_aggregated_features[9,8] ] = 0
 Conv_2_averages_constraint[9,9]: [
   A[0,9] * Conv_2_aggregated_features[9,9]
   + A[1,9] * Conv_2_aggregated_features[9,9]
   + A[2,9] * Conv_2_aggregated_features[9,9]
   + A[3,9] * Conv_2_aggregated_features[9,9]
   + A[4,9] * Conv_2_aggregated_features[9,9]
   + A[5,9] * Conv_2_aggregated_features[9,9]
   + A[6,9] * Conv_2_aggregated_features[9,9]
   + A[7,9] * Conv_2_aggregated_features[9,9]
   + A[8,9] * Conv_2_aggregated_features[9,9]
   - A[9,0] * Conv_1_Relu_ts[0,9] - A[9,1] * Conv_1_Relu_ts[1,9]
   - A[9,2] * Conv_1_Relu_ts[2,9] - A[9,3] * Conv_1_Relu_ts[3,9]
   - A[9,4] * Conv_1_Relu_ts[4,9] - A[9,5] * Conv_1_Relu_ts[5,9]
   - A[9,6] * Conv_1_Relu_ts[6,9] - A[9,7] * Conv_1_Relu_ts[7,9]
   - A[9,8] * Conv_1_Relu_ts[8,9] - A[9,9] * Conv_1_Relu_ts[9,9]
   + A[9,9] * Conv_2_aggregated_features[9,9] ] = 0
 Conv_2_averages_constraint[9,10]: [
   A[0,9] * Conv_2_aggregated_features[9,10]
   + A[1,9] * Conv_2_aggregated_features[9,10]
   + A[2,9] * Conv_2_aggregated_features[9,10]
   + A[3,9] * Conv_2_aggregated_features[9,10]
   + A[4,9] * Conv_2_aggregated_features[9,10]
   + A[5,9] * Conv_2_aggregated_features[9,10]
   + A[6,9] * Conv_2_aggregated_features[9,10]
   + A[7,9] * Conv_2_aggregated_features[9,10]
   + A[8,9] * Conv_2_aggregated_features[9,10]
   - A[9,0] * Conv_1_Relu_ts[0,10] - A[9,1] * Conv_1_Relu_ts[1,10]
   - A[9,2] * Conv_1_Relu_ts[2,10] - A[9,3] * Conv_1_Relu_ts[3,10]
   - A[9,4] * Conv_1_Relu_ts[4,10] - A[9,5] * Conv_1_Relu_ts[5,10]
   - A[9,6] * Conv_1_Relu_ts[6,10] - A[9,7] * Conv_1_Relu_ts[7,10]
   - A[9,8] * Conv_1_Relu_ts[8,10] - A[9,9] * Conv_1_Relu_ts[9,10]
   + A[9,9] * Conv_2_aggregated_features[9,10] ] = 0
 Conv_2_averages_constraint[9,11]: [
   A[0,9] * Conv_2_aggregated_features[9,11]
   + A[1,9] * Conv_2_aggregated_features[9,11]
   + A[2,9] * Conv_2_aggregated_features[9,11]
   + A[3,9] * Conv_2_aggregated_features[9,11]
   + A[4,9] * Conv_2_aggregated_features[9,11]
   + A[5,9] * Conv_2_aggregated_features[9,11]
   + A[6,9] * Conv_2_aggregated_features[9,11]
   + A[7,9] * Conv_2_aggregated_features[9,11]
   + A[8,9] * Conv_2_aggregated_features[9,11]
   - A[9,0] * Conv_1_Relu_ts[0,11] - A[9,1] * Conv_1_Relu_ts[1,11]
   - A[9,2] * Conv_1_Relu_ts[2,11] - A[9,3] * Conv_1_Relu_ts[3,11]
   - A[9,4] * Conv_1_Relu_ts[4,11] - A[9,5] * Conv_1_Relu_ts[5,11]
   - A[9,6] * Conv_1_Relu_ts[6,11] - A[9,7] * Conv_1_Relu_ts[7,11]
   - A[9,8] * Conv_1_Relu_ts[8,11] - A[9,9] * Conv_1_Relu_ts[9,11]
   + A[9,9] * Conv_2_aggregated_features[9,11] ] = 0
 Conv_2_averages_constraint[9,12]: [
   A[0,9] * Conv_2_aggregated_features[9,12]
   + A[1,9] * Conv_2_aggregated_features[9,12]
   + A[2,9] * Conv_2_aggregated_features[9,12]
   + A[3,9] * Conv_2_aggregated_features[9,12]
   + A[4,9] * Conv_2_aggregated_features[9,12]
   + A[5,9] * Conv_2_aggregated_features[9,12]
   + A[6,9] * Conv_2_aggregated_features[9,12]
   + A[7,9] * Conv_2_aggregated_features[9,12]
   + A[8,9] * Conv_2_aggregated_features[9,12]
   - A[9,0] * Conv_1_Relu_ts[0,12] - A[9,1] * Conv_1_Relu_ts[1,12]
   - A[9,2] * Conv_1_Relu_ts[2,12] - A[9,3] * Conv_1_Relu_ts[3,12]
   - A[9,4] * Conv_1_Relu_ts[4,12] - A[9,5] * Conv_1_Relu_ts[5,12]
   - A[9,6] * Conv_1_Relu_ts[6,12] - A[9,7] * Conv_1_Relu_ts[7,12]
   - A[9,8] * Conv_1_Relu_ts[8,12] - A[9,9] * Conv_1_Relu_ts[9,12]
   + A[9,9] * Conv_2_aggregated_features[9,12] ] = 0
 Conv_2_averages_constraint[9,13]: [
   A[0,9] * Conv_2_aggregated_features[9,13]
   + A[1,9] * Conv_2_aggregated_features[9,13]
   + A[2,9] * Conv_2_aggregated_features[9,13]
   + A[3,9] * Conv_2_aggregated_features[9,13]
   + A[4,9] * Conv_2_aggregated_features[9,13]
   + A[5,9] * Conv_2_aggregated_features[9,13]
   + A[6,9] * Conv_2_aggregated_features[9,13]
   + A[7,9] * Conv_2_aggregated_features[9,13]
   + A[8,9] * Conv_2_aggregated_features[9,13]
   - A[9,0] * Conv_1_Relu_ts[0,13] - A[9,1] * Conv_1_Relu_ts[1,13]
   - A[9,2] * Conv_1_Relu_ts[2,13] - A[9,3] * Conv_1_Relu_ts[3,13]
   - A[9,4] * Conv_1_Relu_ts[4,13] - A[9,5] * Conv_1_Relu_ts[5,13]
   - A[9,6] * Conv_1_Relu_ts[6,13] - A[9,7] * Conv_1_Relu_ts[7,13]
   - A[9,8] * Conv_1_Relu_ts[8,13] - A[9,9] * Conv_1_Relu_ts[9,13]
   + A[9,9] * Conv_2_aggregated_features[9,13] ] = 0
 Conv_2_averages_constraint[9,14]: [
   A[0,9] * Conv_2_aggregated_features[9,14]
   + A[1,9] * Conv_2_aggregated_features[9,14]
   + A[2,9] * Conv_2_aggregated_features[9,14]
   + A[3,9] * Conv_2_aggregated_features[9,14]
   + A[4,9] * Conv_2_aggregated_features[9,14]
   + A[5,9] * Conv_2_aggregated_features[9,14]
   + A[6,9] * Conv_2_aggregated_features[9,14]
   + A[7,9] * Conv_2_aggregated_features[9,14]
   + A[8,9] * Conv_2_aggregated_features[9,14]
   - A[9,0] * Conv_1_Relu_ts[0,14] - A[9,1] * Conv_1_Relu_ts[1,14]
   - A[9,2] * Conv_1_Relu_ts[2,14] - A[9,3] * Conv_1_Relu_ts[3,14]
   - A[9,4] * Conv_1_Relu_ts[4,14] - A[9,5] * Conv_1_Relu_ts[5,14]
   - A[9,6] * Conv_1_Relu_ts[6,14] - A[9,7] * Conv_1_Relu_ts[7,14]
   - A[9,8] * Conv_1_Relu_ts[8,14] - A[9,9] * Conv_1_Relu_ts[9,14]
   + A[9,9] * Conv_2_aggregated_features[9,14] ] = 0
 Conv_2_averages_constraint[9,15]: [
   A[0,9] * Conv_2_aggregated_features[9,15]
   + A[1,9] * Conv_2_aggregated_features[9,15]
   + A[2,9] * Conv_2_aggregated_features[9,15]
   + A[3,9] * Conv_2_aggregated_features[9,15]
   + A[4,9] * Conv_2_aggregated_features[9,15]
   + A[5,9] * Conv_2_aggregated_features[9,15]
   + A[6,9] * Conv_2_aggregated_features[9,15]
   + A[7,9] * Conv_2_aggregated_features[9,15]
   + A[8,9] * Conv_2_aggregated_features[9,15]
   - A[9,0] * Conv_1_Relu_ts[0,15] - A[9,1] * Conv_1_Relu_ts[1,15]
   - A[9,2] * Conv_1_Relu_ts[2,15] - A[9,3] * Conv_1_Relu_ts[3,15]
   - A[9,4] * Conv_1_Relu_ts[4,15] - A[9,5] * Conv_1_Relu_ts[5,15]
   - A[9,6] * Conv_1_Relu_ts[6,15] - A[9,7] * Conv_1_Relu_ts[7,15]
   - A[9,8] * Conv_1_Relu_ts[8,15] - A[9,9] * Conv_1_Relu_ts[9,15]
   + A[9,9] * Conv_2_aggregated_features[9,15] ] = 0
 cosine_similarity: 0.5658791065216064 Aggregation[0]
   + 0.000597798 Aggregation[1] + 0.1823571175336838 Aggregation[2]
   + 3.38681173324585 Aggregation[3] + 1.113271594047546 Aggregation[4]
   + 1.020222544670105 Aggregation[5] + 9.390049934387207 Aggregation[6]
   + [ - 10.11322021484375 embedding_similarity * embedding_magnitude ]
   = 0
Bounds
 X[0,0] <= 10
 X[1,0] <= 10
 X[2,0] <= 10
 X[3,0] <= 10
 X[4,0] <= 10
 X[5,0] <= 10
 X[6,0] <= 10
 X[7,0] <= 10
 X[8,0] <= 10
 X[9,0] <= 10
 Conv_0_aggregated_features[0,0] <= 10
 Conv_0_aggregated_features[1,0] <= 10
 Conv_0_aggregated_features[2,0] <= 10
 Conv_0_aggregated_features[3,0] <= 10
 Conv_0_aggregated_features[4,0] <= 10
 Conv_0_aggregated_features[5,0] <= 10
 Conv_0_aggregated_features[6,0] <= 10
 Conv_0_aggregated_features[7,0] <= 10
 Conv_0_aggregated_features[8,0] <= 10
 Conv_0_aggregated_features[9,0] <= 10
 -3.234174191951752 <= Conv_0_t[0,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[0,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[0,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[0,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[0,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[0,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[0,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[0,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[0,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[0,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[0,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[0,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[0,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[0,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[0,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[0,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[1,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[1,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[1,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[1,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[1,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[1,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[1,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[1,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[1,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[1,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[1,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[1,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[1,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[1,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[1,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[1,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[2,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[2,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[2,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[2,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[2,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[2,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[2,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[2,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[2,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[2,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[2,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[2,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[2,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[2,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[2,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[2,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[3,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[3,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[3,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[3,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[3,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[3,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[3,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[3,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[3,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[3,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[3,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[3,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[3,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[3,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[3,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[3,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[4,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[4,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[4,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[4,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[4,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[4,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[4,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[4,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[4,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[4,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[4,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[4,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[4,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[4,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[4,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[4,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[5,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[5,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[5,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[5,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[5,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[5,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[5,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[5,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[5,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[5,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[5,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[5,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[5,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[5,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[5,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[5,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[6,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[6,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[6,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[6,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[6,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[6,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[6,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[6,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[6,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[6,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[6,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[6,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[6,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[6,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[6,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[6,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[7,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[7,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[7,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[7,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[7,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[7,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[7,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[7,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[7,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[7,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[7,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[7,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[7,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[7,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[7,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[7,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[8,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[8,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[8,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[8,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[8,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[8,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[8,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[8,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[8,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[8,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[8,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[8,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[8,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[8,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[8,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[8,15] <= -1.766824722290039
 -3.234174191951752 <= Conv_0_t[9,0] <= -0.8330563721247017
 -11.7988965511322 <= Conv_0_t[9,1] <= 17.59198212623596
 -16.64243040978909 <= Conv_0_t[9,2] <= -1.321228504180908
 -14.8716185092926 <= Conv_0_t[9,3] <= -0.4734632968902588
 -16.71291607618332 <= Conv_0_t[9,4] <= 1.639361500740051
 -15.65530395507812 <= Conv_0_t[9,5] <= 4.801135152578354
 -14.28846871852875 <= Conv_0_t[9,6] <= 2.349466323852539
 -16.01934917271137 <= Conv_0_t[9,7] <= 0.9157499670982361
 -10.0236170142889 <= Conv_0_t[9,8] <= -0.059127077460289
 -12.89834886230528 <= Conv_0_t[9,9] <= -1.259021520614624
 -17.61239869892597 <= Conv_0_t[9,10] <= -1.292696356773376
 -17.77458453178406 <= Conv_0_t[9,11] <= 4.657471835613251
 -6.393570899963379 <= Conv_0_t[9,12] <= 0.0641107559204102
 -8.590249866247177 <= Conv_0_t[9,13] <= 3.006435632705688
 -15.7113471031189 <= Conv_0_t[9,14] <= 4.670293092727661
 -4.252022132277489 <= Conv_0_t[9,15] <= -1.766824722290039
 Conv_0_Relu_ts[0,0] = 0
 Conv_0_Relu_ts[0,1] <= 17.59198212623596
 Conv_0_Relu_ts[0,2] = 0
 Conv_0_Relu_ts[0,3] = 0
 Conv_0_Relu_ts[0,4] <= 1.639361500740051
 Conv_0_Relu_ts[0,5] <= 4.801135152578354
 Conv_0_Relu_ts[0,6] <= 2.349466323852539
 Conv_0_Relu_ts[0,7] <= 0.9157499670982361
 Conv_0_Relu_ts[0,8] = 0
 Conv_0_Relu_ts[0,9] = 0
 Conv_0_Relu_ts[0,10] = 0
 Conv_0_Relu_ts[0,11] <= 4.657471835613251
 Conv_0_Relu_ts[0,12] <= 0.0641107559204102
 Conv_0_Relu_ts[0,13] <= 3.006435632705688
 Conv_0_Relu_ts[0,14] <= 4.670293092727661
 Conv_0_Relu_ts[0,15] = 0
 Conv_0_Relu_ts[1,0] = 0
 Conv_0_Relu_ts[1,1] <= 17.59198212623596
 Conv_0_Relu_ts[1,2] = 0
 Conv_0_Relu_ts[1,3] = 0
 Conv_0_Relu_ts[1,4] <= 1.639361500740051
 Conv_0_Relu_ts[1,5] <= 4.801135152578354
 Conv_0_Relu_ts[1,6] <= 2.349466323852539
 Conv_0_Relu_ts[1,7] <= 0.9157499670982361
 Conv_0_Relu_ts[1,8] = 0
 Conv_0_Relu_ts[1,9] = 0
 Conv_0_Relu_ts[1,10] = 0
 Conv_0_Relu_ts[1,11] <= 4.657471835613251
 Conv_0_Relu_ts[1,12] <= 0.0641107559204102
 Conv_0_Relu_ts[1,13] <= 3.006435632705688
 Conv_0_Relu_ts[1,14] <= 4.670293092727661
 Conv_0_Relu_ts[1,15] = 0
 Conv_0_Relu_ts[2,0] = 0
 Conv_0_Relu_ts[2,1] <= 17.59198212623596
 Conv_0_Relu_ts[2,2] = 0
 Conv_0_Relu_ts[2,3] = 0
 Conv_0_Relu_ts[2,4] <= 1.639361500740051
 Conv_0_Relu_ts[2,5] <= 4.801135152578354
 Conv_0_Relu_ts[2,6] <= 2.349466323852539
 Conv_0_Relu_ts[2,7] <= 0.9157499670982361
 Conv_0_Relu_ts[2,8] = 0
 Conv_0_Relu_ts[2,9] = 0
 Conv_0_Relu_ts[2,10] = 0
 Conv_0_Relu_ts[2,11] <= 4.657471835613251
 Conv_0_Relu_ts[2,12] <= 0.0641107559204102
 Conv_0_Relu_ts[2,13] <= 3.006435632705688
 Conv_0_Relu_ts[2,14] <= 4.670293092727661
 Conv_0_Relu_ts[2,15] = 0
 Conv_0_Relu_ts[3,0] = 0
 Conv_0_Relu_ts[3,1] <= 17.59198212623596
 Conv_0_Relu_ts[3,2] = 0
 Conv_0_Relu_ts[3,3] = 0
 Conv_0_Relu_ts[3,4] <= 1.639361500740051
 Conv_0_Relu_ts[3,5] <= 4.801135152578354
 Conv_0_Relu_ts[3,6] <= 2.349466323852539
 Conv_0_Relu_ts[3,7] <= 0.9157499670982361
 Conv_0_Relu_ts[3,8] = 0
 Conv_0_Relu_ts[3,9] = 0
 Conv_0_Relu_ts[3,10] = 0
 Conv_0_Relu_ts[3,11] <= 4.657471835613251
 Conv_0_Relu_ts[3,12] <= 0.0641107559204102
 Conv_0_Relu_ts[3,13] <= 3.006435632705688
 Conv_0_Relu_ts[3,14] <= 4.670293092727661
 Conv_0_Relu_ts[3,15] = 0
 Conv_0_Relu_ts[4,0] = 0
 Conv_0_Relu_ts[4,1] <= 17.59198212623596
 Conv_0_Relu_ts[4,2] = 0
 Conv_0_Relu_ts[4,3] = 0
 Conv_0_Relu_ts[4,4] <= 1.639361500740051
 Conv_0_Relu_ts[4,5] <= 4.801135152578354
 Conv_0_Relu_ts[4,6] <= 2.349466323852539
 Conv_0_Relu_ts[4,7] <= 0.9157499670982361
 Conv_0_Relu_ts[4,8] = 0
 Conv_0_Relu_ts[4,9] = 0
 Conv_0_Relu_ts[4,10] = 0
 Conv_0_Relu_ts[4,11] <= 4.657471835613251
 Conv_0_Relu_ts[4,12] <= 0.0641107559204102
 Conv_0_Relu_ts[4,13] <= 3.006435632705688
 Conv_0_Relu_ts[4,14] <= 4.670293092727661
 Conv_0_Relu_ts[4,15] = 0
 Conv_0_Relu_ts[5,0] = 0
 Conv_0_Relu_ts[5,1] <= 17.59198212623596
 Conv_0_Relu_ts[5,2] = 0
 Conv_0_Relu_ts[5,3] = 0
 Conv_0_Relu_ts[5,4] <= 1.639361500740051
 Conv_0_Relu_ts[5,5] <= 4.801135152578354
 Conv_0_Relu_ts[5,6] <= 2.349466323852539
 Conv_0_Relu_ts[5,7] <= 0.9157499670982361
 Conv_0_Relu_ts[5,8] = 0
 Conv_0_Relu_ts[5,9] = 0
 Conv_0_Relu_ts[5,10] = 0
 Conv_0_Relu_ts[5,11] <= 4.657471835613251
 Conv_0_Relu_ts[5,12] <= 0.0641107559204102
 Conv_0_Relu_ts[5,13] <= 3.006435632705688
 Conv_0_Relu_ts[5,14] <= 4.670293092727661
 Conv_0_Relu_ts[5,15] = 0
 Conv_0_Relu_ts[6,0] = 0
 Conv_0_Relu_ts[6,1] <= 17.59198212623596
 Conv_0_Relu_ts[6,2] = 0
 Conv_0_Relu_ts[6,3] = 0
 Conv_0_Relu_ts[6,4] <= 1.639361500740051
 Conv_0_Relu_ts[6,5] <= 4.801135152578354
 Conv_0_Relu_ts[6,6] <= 2.349466323852539
 Conv_0_Relu_ts[6,7] <= 0.9157499670982361
 Conv_0_Relu_ts[6,8] = 0
 Conv_0_Relu_ts[6,9] = 0
 Conv_0_Relu_ts[6,10] = 0
 Conv_0_Relu_ts[6,11] <= 4.657471835613251
 Conv_0_Relu_ts[6,12] <= 0.0641107559204102
 Conv_0_Relu_ts[6,13] <= 3.006435632705688
 Conv_0_Relu_ts[6,14] <= 4.670293092727661
 Conv_0_Relu_ts[6,15] = 0
 Conv_0_Relu_ts[7,0] = 0
 Conv_0_Relu_ts[7,1] <= 17.59198212623596
 Conv_0_Relu_ts[7,2] = 0
 Conv_0_Relu_ts[7,3] = 0
 Conv_0_Relu_ts[7,4] <= 1.639361500740051
 Conv_0_Relu_ts[7,5] <= 4.801135152578354
 Conv_0_Relu_ts[7,6] <= 2.349466323852539
 Conv_0_Relu_ts[7,7] <= 0.9157499670982361
 Conv_0_Relu_ts[7,8] = 0
 Conv_0_Relu_ts[7,9] = 0
 Conv_0_Relu_ts[7,10] = 0
 Conv_0_Relu_ts[7,11] <= 4.657471835613251
 Conv_0_Relu_ts[7,12] <= 0.0641107559204102
 Conv_0_Relu_ts[7,13] <= 3.006435632705688
 Conv_0_Relu_ts[7,14] <= 4.670293092727661
 Conv_0_Relu_ts[7,15] = 0
 Conv_0_Relu_ts[8,0] = 0
 Conv_0_Relu_ts[8,1] <= 17.59198212623596
 Conv_0_Relu_ts[8,2] = 0
 Conv_0_Relu_ts[8,3] = 0
 Conv_0_Relu_ts[8,4] <= 1.639361500740051
 Conv_0_Relu_ts[8,5] <= 4.801135152578354
 Conv_0_Relu_ts[8,6] <= 2.349466323852539
 Conv_0_Relu_ts[8,7] <= 0.9157499670982361
 Conv_0_Relu_ts[8,8] = 0
 Conv_0_Relu_ts[8,9] = 0
 Conv_0_Relu_ts[8,10] = 0
 Conv_0_Relu_ts[8,11] <= 4.657471835613251
 Conv_0_Relu_ts[8,12] <= 0.0641107559204102
 Conv_0_Relu_ts[8,13] <= 3.006435632705688
 Conv_0_Relu_ts[8,14] <= 4.670293092727661
 Conv_0_Relu_ts[8,15] = 0
 Conv_0_Relu_ts[9,0] = 0
 Conv_0_Relu_ts[9,1] <= 17.59198212623596
 Conv_0_Relu_ts[9,2] = 0
 Conv_0_Relu_ts[9,3] = 0
 Conv_0_Relu_ts[9,4] <= 1.639361500740051
 Conv_0_Relu_ts[9,5] <= 4.801135152578354
 Conv_0_Relu_ts[9,6] <= 2.349466323852539
 Conv_0_Relu_ts[9,7] <= 0.9157499670982361
 Conv_0_Relu_ts[9,8] = 0
 Conv_0_Relu_ts[9,9] = 0
 Conv_0_Relu_ts[9,10] = 0
 Conv_0_Relu_ts[9,11] <= 4.657471835613251
 Conv_0_Relu_ts[9,12] <= 0.0641107559204102
 Conv_0_Relu_ts[9,13] <= 3.006435632705688
 Conv_0_Relu_ts[9,14] <= 4.670293092727661
 Conv_0_Relu_ts[9,15] = 0
 Conv_0_Relu_ss[0,0] <= 3.234174191951752
 Conv_0_Relu_ss[0,1] <= 11.7988965511322
 Conv_0_Relu_ss[0,2] <= 16.64243040978909
 Conv_0_Relu_ss[0,3] <= 14.8716185092926
 Conv_0_Relu_ss[0,4] <= 16.71291607618332
 Conv_0_Relu_ss[0,5] <= 15.65530395507812
 Conv_0_Relu_ss[0,6] <= 14.28846871852875
 Conv_0_Relu_ss[0,7] <= 16.01934917271137
 Conv_0_Relu_ss[0,8] <= 10.0236170142889
 Conv_0_Relu_ss[0,9] <= 12.89834886230528
 Conv_0_Relu_ss[0,10] <= 17.61239869892597
 Conv_0_Relu_ss[0,11] <= 17.77458453178406
 Conv_0_Relu_ss[0,12] <= 6.393570899963379
 Conv_0_Relu_ss[0,13] <= 8.590249866247177
 Conv_0_Relu_ss[0,14] <= 15.7113471031189
 Conv_0_Relu_ss[0,15] <= 4.252022132277489
 Conv_0_Relu_ss[1,0] <= 3.234174191951752
 Conv_0_Relu_ss[1,1] <= 11.7988965511322
 Conv_0_Relu_ss[1,2] <= 16.64243040978909
 Conv_0_Relu_ss[1,3] <= 14.8716185092926
 Conv_0_Relu_ss[1,4] <= 16.71291607618332
 Conv_0_Relu_ss[1,5] <= 15.65530395507812
 Conv_0_Relu_ss[1,6] <= 14.28846871852875
 Conv_0_Relu_ss[1,7] <= 16.01934917271137
 Conv_0_Relu_ss[1,8] <= 10.0236170142889
 Conv_0_Relu_ss[1,9] <= 12.89834886230528
 Conv_0_Relu_ss[1,10] <= 17.61239869892597
 Conv_0_Relu_ss[1,11] <= 17.77458453178406
 Conv_0_Relu_ss[1,12] <= 6.393570899963379
 Conv_0_Relu_ss[1,13] <= 8.590249866247177
 Conv_0_Relu_ss[1,14] <= 15.7113471031189
 Conv_0_Relu_ss[1,15] <= 4.252022132277489
 Conv_0_Relu_ss[2,0] <= 3.234174191951752
 Conv_0_Relu_ss[2,1] <= 11.7988965511322
 Conv_0_Relu_ss[2,2] <= 16.64243040978909
 Conv_0_Relu_ss[2,3] <= 14.8716185092926
 Conv_0_Relu_ss[2,4] <= 16.71291607618332
 Conv_0_Relu_ss[2,5] <= 15.65530395507812
 Conv_0_Relu_ss[2,6] <= 14.28846871852875
 Conv_0_Relu_ss[2,7] <= 16.01934917271137
 Conv_0_Relu_ss[2,8] <= 10.0236170142889
 Conv_0_Relu_ss[2,9] <= 12.89834886230528
 Conv_0_Relu_ss[2,10] <= 17.61239869892597
 Conv_0_Relu_ss[2,11] <= 17.77458453178406
 Conv_0_Relu_ss[2,12] <= 6.393570899963379
 Conv_0_Relu_ss[2,13] <= 8.590249866247177
 Conv_0_Relu_ss[2,14] <= 15.7113471031189
 Conv_0_Relu_ss[2,15] <= 4.252022132277489
 Conv_0_Relu_ss[3,0] <= 3.234174191951752
 Conv_0_Relu_ss[3,1] <= 11.7988965511322
 Conv_0_Relu_ss[3,2] <= 16.64243040978909
 Conv_0_Relu_ss[3,3] <= 14.8716185092926
 Conv_0_Relu_ss[3,4] <= 16.71291607618332
 Conv_0_Relu_ss[3,5] <= 15.65530395507812
 Conv_0_Relu_ss[3,6] <= 14.28846871852875
 Conv_0_Relu_ss[3,7] <= 16.01934917271137
 Conv_0_Relu_ss[3,8] <= 10.0236170142889
 Conv_0_Relu_ss[3,9] <= 12.89834886230528
 Conv_0_Relu_ss[3,10] <= 17.61239869892597
 Conv_0_Relu_ss[3,11] <= 17.77458453178406
 Conv_0_Relu_ss[3,12] <= 6.393570899963379
 Conv_0_Relu_ss[3,13] <= 8.590249866247177
 Conv_0_Relu_ss[3,14] <= 15.7113471031189
 Conv_0_Relu_ss[3,15] <= 4.252022132277489
 Conv_0_Relu_ss[4,0] <= 3.234174191951752
 Conv_0_Relu_ss[4,1] <= 11.7988965511322
 Conv_0_Relu_ss[4,2] <= 16.64243040978909
 Conv_0_Relu_ss[4,3] <= 14.8716185092926
 Conv_0_Relu_ss[4,4] <= 16.71291607618332
 Conv_0_Relu_ss[4,5] <= 15.65530395507812
 Conv_0_Relu_ss[4,6] <= 14.28846871852875
 Conv_0_Relu_ss[4,7] <= 16.01934917271137
 Conv_0_Relu_ss[4,8] <= 10.0236170142889
 Conv_0_Relu_ss[4,9] <= 12.89834886230528
 Conv_0_Relu_ss[4,10] <= 17.61239869892597
 Conv_0_Relu_ss[4,11] <= 17.77458453178406
 Conv_0_Relu_ss[4,12] <= 6.393570899963379
 Conv_0_Relu_ss[4,13] <= 8.590249866247177
 Conv_0_Relu_ss[4,14] <= 15.7113471031189
 Conv_0_Relu_ss[4,15] <= 4.252022132277489
 Conv_0_Relu_ss[5,0] <= 3.234174191951752
 Conv_0_Relu_ss[5,1] <= 11.7988965511322
 Conv_0_Relu_ss[5,2] <= 16.64243040978909
 Conv_0_Relu_ss[5,3] <= 14.8716185092926
 Conv_0_Relu_ss[5,4] <= 16.71291607618332
 Conv_0_Relu_ss[5,5] <= 15.65530395507812
 Conv_0_Relu_ss[5,6] <= 14.28846871852875
 Conv_0_Relu_ss[5,7] <= 16.01934917271137
 Conv_0_Relu_ss[5,8] <= 10.0236170142889
 Conv_0_Relu_ss[5,9] <= 12.89834886230528
 Conv_0_Relu_ss[5,10] <= 17.61239869892597
 Conv_0_Relu_ss[5,11] <= 17.77458453178406
 Conv_0_Relu_ss[5,12] <= 6.393570899963379
 Conv_0_Relu_ss[5,13] <= 8.590249866247177
 Conv_0_Relu_ss[5,14] <= 15.7113471031189
 Conv_0_Relu_ss[5,15] <= 4.252022132277489
 Conv_0_Relu_ss[6,0] <= 3.234174191951752
 Conv_0_Relu_ss[6,1] <= 11.7988965511322
 Conv_0_Relu_ss[6,2] <= 16.64243040978909
 Conv_0_Relu_ss[6,3] <= 14.8716185092926
 Conv_0_Relu_ss[6,4] <= 16.71291607618332
 Conv_0_Relu_ss[6,5] <= 15.65530395507812
 Conv_0_Relu_ss[6,6] <= 14.28846871852875
 Conv_0_Relu_ss[6,7] <= 16.01934917271137
 Conv_0_Relu_ss[6,8] <= 10.0236170142889
 Conv_0_Relu_ss[6,9] <= 12.89834886230528
 Conv_0_Relu_ss[6,10] <= 17.61239869892597
 Conv_0_Relu_ss[6,11] <= 17.77458453178406
 Conv_0_Relu_ss[6,12] <= 6.393570899963379
 Conv_0_Relu_ss[6,13] <= 8.590249866247177
 Conv_0_Relu_ss[6,14] <= 15.7113471031189
 Conv_0_Relu_ss[6,15] <= 4.252022132277489
 Conv_0_Relu_ss[7,0] <= 3.234174191951752
 Conv_0_Relu_ss[7,1] <= 11.7988965511322
 Conv_0_Relu_ss[7,2] <= 16.64243040978909
 Conv_0_Relu_ss[7,3] <= 14.8716185092926
 Conv_0_Relu_ss[7,4] <= 16.71291607618332
 Conv_0_Relu_ss[7,5] <= 15.65530395507812
 Conv_0_Relu_ss[7,6] <= 14.28846871852875
 Conv_0_Relu_ss[7,7] <= 16.01934917271137
 Conv_0_Relu_ss[7,8] <= 10.0236170142889
 Conv_0_Relu_ss[7,9] <= 12.89834886230528
 Conv_0_Relu_ss[7,10] <= 17.61239869892597
 Conv_0_Relu_ss[7,11] <= 17.77458453178406
 Conv_0_Relu_ss[7,12] <= 6.393570899963379
 Conv_0_Relu_ss[7,13] <= 8.590249866247177
 Conv_0_Relu_ss[7,14] <= 15.7113471031189
 Conv_0_Relu_ss[7,15] <= 4.252022132277489
 Conv_0_Relu_ss[8,0] <= 3.234174191951752
 Conv_0_Relu_ss[8,1] <= 11.7988965511322
 Conv_0_Relu_ss[8,2] <= 16.64243040978909
 Conv_0_Relu_ss[8,3] <= 14.8716185092926
 Conv_0_Relu_ss[8,4] <= 16.71291607618332
 Conv_0_Relu_ss[8,5] <= 15.65530395507812
 Conv_0_Relu_ss[8,6] <= 14.28846871852875
 Conv_0_Relu_ss[8,7] <= 16.01934917271137
 Conv_0_Relu_ss[8,8] <= 10.0236170142889
 Conv_0_Relu_ss[8,9] <= 12.89834886230528
 Conv_0_Relu_ss[8,10] <= 17.61239869892597
 Conv_0_Relu_ss[8,11] <= 17.77458453178406
 Conv_0_Relu_ss[8,12] <= 6.393570899963379
 Conv_0_Relu_ss[8,13] <= 8.590249866247177
 Conv_0_Relu_ss[8,14] <= 15.7113471031189
 Conv_0_Relu_ss[8,15] <= 4.252022132277489
 Conv_0_Relu_ss[9,0] <= 3.234174191951752
 Conv_0_Relu_ss[9,1] <= 11.7988965511322
 Conv_0_Relu_ss[9,2] <= 16.64243040978909
 Conv_0_Relu_ss[9,3] <= 14.8716185092926
 Conv_0_Relu_ss[9,4] <= 16.71291607618332
 Conv_0_Relu_ss[9,5] <= 15.65530395507812
 Conv_0_Relu_ss[9,6] <= 14.28846871852875
 Conv_0_Relu_ss[9,7] <= 16.01934917271137
 Conv_0_Relu_ss[9,8] <= 10.0236170142889
 Conv_0_Relu_ss[9,9] <= 12.89834886230528
 Conv_0_Relu_ss[9,10] <= 17.61239869892597
 Conv_0_Relu_ss[9,11] <= 17.77458453178406
 Conv_0_Relu_ss[9,12] <= 6.393570899963379
 Conv_0_Relu_ss[9,13] <= 8.590249866247177
 Conv_0_Relu_ss[9,14] <= 15.7113471031189
 Conv_0_Relu_ss[9,15] <= 4.252022132277489
 Conv_1_aggregated_features[0,0] = 0
 Conv_1_aggregated_features[0,1] <= 17.59198212623596
 Conv_1_aggregated_features[0,2] = 0
 Conv_1_aggregated_features[0,3] = 0
 Conv_1_aggregated_features[0,4] <= 1.639361500740051
 Conv_1_aggregated_features[0,5] <= 4.801135152578354
 Conv_1_aggregated_features[0,6] <= 2.349466323852539
 Conv_1_aggregated_features[0,7] <= 0.9157499670982361
 Conv_1_aggregated_features[0,8] = 0
 Conv_1_aggregated_features[0,9] = 0
 Conv_1_aggregated_features[0,10] = 0
 Conv_1_aggregated_features[0,11] <= 4.657471835613251
 Conv_1_aggregated_features[0,12] <= 0.0641107559204102
 Conv_1_aggregated_features[0,13] <= 3.006435632705688
 Conv_1_aggregated_features[0,14] <= 4.670293092727661
 Conv_1_aggregated_features[0,15] = 0
 Conv_1_aggregated_features[1,0] = 0
 Conv_1_aggregated_features[1,1] <= 17.59198212623596
 Conv_1_aggregated_features[1,2] = 0
 Conv_1_aggregated_features[1,3] = 0
 Conv_1_aggregated_features[1,4] <= 1.639361500740051
 Conv_1_aggregated_features[1,5] <= 4.801135152578354
 Conv_1_aggregated_features[1,6] <= 2.349466323852539
 Conv_1_aggregated_features[1,7] <= 0.9157499670982361
 Conv_1_aggregated_features[1,8] = 0
 Conv_1_aggregated_features[1,9] = 0
 Conv_1_aggregated_features[1,10] = 0
 Conv_1_aggregated_features[1,11] <= 4.657471835613251
 Conv_1_aggregated_features[1,12] <= 0.0641107559204102
 Conv_1_aggregated_features[1,13] <= 3.006435632705688
 Conv_1_aggregated_features[1,14] <= 4.670293092727661
 Conv_1_aggregated_features[1,15] = 0
 Conv_1_aggregated_features[2,0] = 0
 Conv_1_aggregated_features[2,1] <= 17.59198212623596
 Conv_1_aggregated_features[2,2] = 0
 Conv_1_aggregated_features[2,3] = 0
 Conv_1_aggregated_features[2,4] <= 1.639361500740051
 Conv_1_aggregated_features[2,5] <= 4.801135152578354
 Conv_1_aggregated_features[2,6] <= 2.349466323852539
 Conv_1_aggregated_features[2,7] <= 0.9157499670982361
 Conv_1_aggregated_features[2,8] = 0
 Conv_1_aggregated_features[2,9] = 0
 Conv_1_aggregated_features[2,10] = 0
 Conv_1_aggregated_features[2,11] <= 4.657471835613251
 Conv_1_aggregated_features[2,12] <= 0.0641107559204102
 Conv_1_aggregated_features[2,13] <= 3.006435632705688
 Conv_1_aggregated_features[2,14] <= 4.670293092727661
 Conv_1_aggregated_features[2,15] = 0
 Conv_1_aggregated_features[3,0] = 0
 Conv_1_aggregated_features[3,1] <= 17.59198212623596
 Conv_1_aggregated_features[3,2] = 0
 Conv_1_aggregated_features[3,3] = 0
 Conv_1_aggregated_features[3,4] <= 1.639361500740051
 Conv_1_aggregated_features[3,5] <= 4.801135152578354
 Conv_1_aggregated_features[3,6] <= 2.349466323852539
 Conv_1_aggregated_features[3,7] <= 0.9157499670982361
 Conv_1_aggregated_features[3,8] = 0
 Conv_1_aggregated_features[3,9] = 0
 Conv_1_aggregated_features[3,10] = 0
 Conv_1_aggregated_features[3,11] <= 4.657471835613251
 Conv_1_aggregated_features[3,12] <= 0.0641107559204102
 Conv_1_aggregated_features[3,13] <= 3.006435632705688
 Conv_1_aggregated_features[3,14] <= 4.670293092727661
 Conv_1_aggregated_features[3,15] = 0
 Conv_1_aggregated_features[4,0] = 0
 Conv_1_aggregated_features[4,1] <= 17.59198212623596
 Conv_1_aggregated_features[4,2] = 0
 Conv_1_aggregated_features[4,3] = 0
 Conv_1_aggregated_features[4,4] <= 1.639361500740051
 Conv_1_aggregated_features[4,5] <= 4.801135152578354
 Conv_1_aggregated_features[4,6] <= 2.349466323852539
 Conv_1_aggregated_features[4,7] <= 0.9157499670982361
 Conv_1_aggregated_features[4,8] = 0
 Conv_1_aggregated_features[4,9] = 0
 Conv_1_aggregated_features[4,10] = 0
 Conv_1_aggregated_features[4,11] <= 4.657471835613251
 Conv_1_aggregated_features[4,12] <= 0.0641107559204102
 Conv_1_aggregated_features[4,13] <= 3.006435632705688
 Conv_1_aggregated_features[4,14] <= 4.670293092727661
 Conv_1_aggregated_features[4,15] = 0
 Conv_1_aggregated_features[5,0] = 0
 Conv_1_aggregated_features[5,1] <= 17.59198212623596
 Conv_1_aggregated_features[5,2] = 0
 Conv_1_aggregated_features[5,3] = 0
 Conv_1_aggregated_features[5,4] <= 1.639361500740051
 Conv_1_aggregated_features[5,5] <= 4.801135152578354
 Conv_1_aggregated_features[5,6] <= 2.349466323852539
 Conv_1_aggregated_features[5,7] <= 0.9157499670982361
 Conv_1_aggregated_features[5,8] = 0
 Conv_1_aggregated_features[5,9] = 0
 Conv_1_aggregated_features[5,10] = 0
 Conv_1_aggregated_features[5,11] <= 4.657471835613251
 Conv_1_aggregated_features[5,12] <= 0.0641107559204102
 Conv_1_aggregated_features[5,13] <= 3.006435632705688
 Conv_1_aggregated_features[5,14] <= 4.670293092727661
 Conv_1_aggregated_features[5,15] = 0
 Conv_1_aggregated_features[6,0] = 0
 Conv_1_aggregated_features[6,1] <= 17.59198212623596
 Conv_1_aggregated_features[6,2] = 0
 Conv_1_aggregated_features[6,3] = 0
 Conv_1_aggregated_features[6,4] <= 1.639361500740051
 Conv_1_aggregated_features[6,5] <= 4.801135152578354
 Conv_1_aggregated_features[6,6] <= 2.349466323852539
 Conv_1_aggregated_features[6,7] <= 0.9157499670982361
 Conv_1_aggregated_features[6,8] = 0
 Conv_1_aggregated_features[6,9] = 0
 Conv_1_aggregated_features[6,10] = 0
 Conv_1_aggregated_features[6,11] <= 4.657471835613251
 Conv_1_aggregated_features[6,12] <= 0.0641107559204102
 Conv_1_aggregated_features[6,13] <= 3.006435632705688
 Conv_1_aggregated_features[6,14] <= 4.670293092727661
 Conv_1_aggregated_features[6,15] = 0
 Conv_1_aggregated_features[7,0] = 0
 Conv_1_aggregated_features[7,1] <= 17.59198212623596
 Conv_1_aggregated_features[7,2] = 0
 Conv_1_aggregated_features[7,3] = 0
 Conv_1_aggregated_features[7,4] <= 1.639361500740051
 Conv_1_aggregated_features[7,5] <= 4.801135152578354
 Conv_1_aggregated_features[7,6] <= 2.349466323852539
 Conv_1_aggregated_features[7,7] <= 0.9157499670982361
 Conv_1_aggregated_features[7,8] = 0
 Conv_1_aggregated_features[7,9] = 0
 Conv_1_aggregated_features[7,10] = 0
 Conv_1_aggregated_features[7,11] <= 4.657471835613251
 Conv_1_aggregated_features[7,12] <= 0.0641107559204102
 Conv_1_aggregated_features[7,13] <= 3.006435632705688
 Conv_1_aggregated_features[7,14] <= 4.670293092727661
 Conv_1_aggregated_features[7,15] = 0
 Conv_1_aggregated_features[8,0] = 0
 Conv_1_aggregated_features[8,1] <= 17.59198212623596
 Conv_1_aggregated_features[8,2] = 0
 Conv_1_aggregated_features[8,3] = 0
 Conv_1_aggregated_features[8,4] <= 1.639361500740051
 Conv_1_aggregated_features[8,5] <= 4.801135152578354
 Conv_1_aggregated_features[8,6] <= 2.349466323852539
 Conv_1_aggregated_features[8,7] <= 0.9157499670982361
 Conv_1_aggregated_features[8,8] = 0
 Conv_1_aggregated_features[8,9] = 0
 Conv_1_aggregated_features[8,10] = 0
 Conv_1_aggregated_features[8,11] <= 4.657471835613251
 Conv_1_aggregated_features[8,12] <= 0.0641107559204102
 Conv_1_aggregated_features[8,13] <= 3.006435632705688
 Conv_1_aggregated_features[8,14] <= 4.670293092727661
 Conv_1_aggregated_features[8,15] = 0
 Conv_1_aggregated_features[9,0] = 0
 Conv_1_aggregated_features[9,1] <= 17.59198212623596
 Conv_1_aggregated_features[9,2] = 0
 Conv_1_aggregated_features[9,3] = 0
 Conv_1_aggregated_features[9,4] <= 1.639361500740051
 Conv_1_aggregated_features[9,5] <= 4.801135152578354
 Conv_1_aggregated_features[9,6] <= 2.349466323852539
 Conv_1_aggregated_features[9,7] <= 0.9157499670982361
 Conv_1_aggregated_features[9,8] = 0
 Conv_1_aggregated_features[9,9] = 0
 Conv_1_aggregated_features[9,10] = 0
 Conv_1_aggregated_features[9,11] <= 4.657471835613251
 Conv_1_aggregated_features[9,12] <= 0.0641107559204102
 Conv_1_aggregated_features[9,13] <= 3.006435632705688
 Conv_1_aggregated_features[9,14] <= 4.670293092727661
 Conv_1_aggregated_features[9,15] = 0
 -56.29245011536772 <= Conv_1_t[0,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[0,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[0,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[0,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[0,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[0,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[0,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[0,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[0,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[0,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[0,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[0,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[0,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[0,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[0,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[0,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[1,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[1,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[1,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[1,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[1,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[1,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[1,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[1,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[1,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[1,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[1,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[1,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[1,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[1,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[1,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[1,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[2,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[2,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[2,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[2,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[2,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[2,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[2,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[2,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[2,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[2,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[2,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[2,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[2,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[2,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[2,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[2,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[3,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[3,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[3,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[3,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[3,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[3,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[3,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[3,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[3,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[3,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[3,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[3,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[3,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[3,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[3,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[3,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[4,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[4,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[4,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[4,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[4,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[4,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[4,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[4,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[4,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[4,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[4,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[4,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[4,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[4,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[4,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[4,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[5,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[5,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[5,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[5,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[5,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[5,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[5,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[5,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[5,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[5,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[5,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[5,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[5,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[5,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[5,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[5,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[6,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[6,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[6,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[6,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[6,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[6,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[6,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[6,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[6,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[6,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[6,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[6,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[6,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[6,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[6,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[6,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[7,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[7,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[7,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[7,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[7,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[7,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[7,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[7,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[7,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[7,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[7,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[7,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[7,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[7,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[7,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[7,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[8,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[8,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[8,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[8,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[8,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[8,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[8,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[8,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[8,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[8,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[8,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[8,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[8,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[8,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[8,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[8,15] <= 1.234436565832815
 -56.29245011536772 <= Conv_1_t[9,0] <= 25.57325229459289
 -67.52969714306539 <= Conv_1_t[9,1] <= 16.63501010363109
 -88.95482652838119 <= Conv_1_t[9,2] <= 35.00484918336429
 -33.94596106828702 <= Conv_1_t[9,3] <= 0.4998085927964429
 -106.1251358090966 <= Conv_1_t[9,4] <= 57.82216539781905
 -10.77560081932031 <= Conv_1_t[9,5] <= 2.411270213327107
 -29.32218285779035 <= Conv_1_t[9,6] <= 49.46156674751891
 -24.30472028988515 <= Conv_1_t[9,7] <= 3.462243420878994
 -25.8812298511837 <= Conv_1_t[9,8] <= 36.11140265688224
 -76.75665981089216 <= Conv_1_t[9,9] <= 9.030128828439624
 -54.2754182075725 <= Conv_1_t[9,10] <= 23.63937264816158
 -97.18723355279275 <= Conv_1_t[9,11] <= 60.18996818642027
 -47.62093859518428 <= Conv_1_t[9,12] <= 29.32487877957098
 -21.10965662228852 <= Conv_1_t[9,13] <= 3.004389269438119
 -21.61040402775194 <= Conv_1_t[9,14] <= 24.37010483280744
 -11.46170062280965 <= Conv_1_t[9,15] <= 1.234436565832815
 Conv_1_Relu_ts[0,0] <= 25.57325229459289
 Conv_1_Relu_ts[0,1] <= 16.63501010363109
 Conv_1_Relu_ts[0,2] <= 35.00484918336429
 Conv_1_Relu_ts[0,3] <= 0.4998085927964429
 Conv_1_Relu_ts[0,4] <= 57.82216539781905
 Conv_1_Relu_ts[0,5] <= 2.411270213327107
 Conv_1_Relu_ts[0,6] <= 49.46156674751891
 Conv_1_Relu_ts[0,7] <= 3.462243420878994
 Conv_1_Relu_ts[0,8] <= 36.11140265688224
 Conv_1_Relu_ts[0,9] <= 9.030128828439624
 Conv_1_Relu_ts[0,10] <= 23.63937264816158
 Conv_1_Relu_ts[0,11] <= 60.18996818642027
 Conv_1_Relu_ts[0,12] <= 29.32487877957098
 Conv_1_Relu_ts[0,13] <= 3.004389269438119
 Conv_1_Relu_ts[0,14] <= 24.37010483280744
 Conv_1_Relu_ts[0,15] <= 1.234436565832815
 Conv_1_Relu_ts[1,0] <= 25.57325229459289
 Conv_1_Relu_ts[1,1] <= 16.63501010363109
 Conv_1_Relu_ts[1,2] <= 35.00484918336429
 Conv_1_Relu_ts[1,3] <= 0.4998085927964429
 Conv_1_Relu_ts[1,4] <= 57.82216539781905
 Conv_1_Relu_ts[1,5] <= 2.411270213327107
 Conv_1_Relu_ts[1,6] <= 49.46156674751891
 Conv_1_Relu_ts[1,7] <= 3.462243420878994
 Conv_1_Relu_ts[1,8] <= 36.11140265688224
 Conv_1_Relu_ts[1,9] <= 9.030128828439624
 Conv_1_Relu_ts[1,10] <= 23.63937264816158
 Conv_1_Relu_ts[1,11] <= 60.18996818642027
 Conv_1_Relu_ts[1,12] <= 29.32487877957098
 Conv_1_Relu_ts[1,13] <= 3.004389269438119
 Conv_1_Relu_ts[1,14] <= 24.37010483280744
 Conv_1_Relu_ts[1,15] <= 1.234436565832815
 Conv_1_Relu_ts[2,0] <= 25.57325229459289
 Conv_1_Relu_ts[2,1] <= 16.63501010363109
 Conv_1_Relu_ts[2,2] <= 35.00484918336429
 Conv_1_Relu_ts[2,3] <= 0.4998085927964429
 Conv_1_Relu_ts[2,4] <= 57.82216539781905
 Conv_1_Relu_ts[2,5] <= 2.411270213327107
 Conv_1_Relu_ts[2,6] <= 49.46156674751891
 Conv_1_Relu_ts[2,7] <= 3.462243420878994
 Conv_1_Relu_ts[2,8] <= 36.11140265688224
 Conv_1_Relu_ts[2,9] <= 9.030128828439624
 Conv_1_Relu_ts[2,10] <= 23.63937264816158
 Conv_1_Relu_ts[2,11] <= 60.18996818642027
 Conv_1_Relu_ts[2,12] <= 29.32487877957098
 Conv_1_Relu_ts[2,13] <= 3.004389269438119
 Conv_1_Relu_ts[2,14] <= 24.37010483280744
 Conv_1_Relu_ts[2,15] <= 1.234436565832815
 Conv_1_Relu_ts[3,0] <= 25.57325229459289
 Conv_1_Relu_ts[3,1] <= 16.63501010363109
 Conv_1_Relu_ts[3,2] <= 35.00484918336429
 Conv_1_Relu_ts[3,3] <= 0.4998085927964429
 Conv_1_Relu_ts[3,4] <= 57.82216539781905
 Conv_1_Relu_ts[3,5] <= 2.411270213327107
 Conv_1_Relu_ts[3,6] <= 49.46156674751891
 Conv_1_Relu_ts[3,7] <= 3.462243420878994
 Conv_1_Relu_ts[3,8] <= 36.11140265688224
 Conv_1_Relu_ts[3,9] <= 9.030128828439624
 Conv_1_Relu_ts[3,10] <= 23.63937264816158
 Conv_1_Relu_ts[3,11] <= 60.18996818642027
 Conv_1_Relu_ts[3,12] <= 29.32487877957098
 Conv_1_Relu_ts[3,13] <= 3.004389269438119
 Conv_1_Relu_ts[3,14] <= 24.37010483280744
 Conv_1_Relu_ts[3,15] <= 1.234436565832815
 Conv_1_Relu_ts[4,0] <= 25.57325229459289
 Conv_1_Relu_ts[4,1] <= 16.63501010363109
 Conv_1_Relu_ts[4,2] <= 35.00484918336429
 Conv_1_Relu_ts[4,3] <= 0.4998085927964429
 Conv_1_Relu_ts[4,4] <= 57.82216539781905
 Conv_1_Relu_ts[4,5] <= 2.411270213327107
 Conv_1_Relu_ts[4,6] <= 49.46156674751891
 Conv_1_Relu_ts[4,7] <= 3.462243420878994
 Conv_1_Relu_ts[4,8] <= 36.11140265688224
 Conv_1_Relu_ts[4,9] <= 9.030128828439624
 Conv_1_Relu_ts[4,10] <= 23.63937264816158
 Conv_1_Relu_ts[4,11] <= 60.18996818642027
 Conv_1_Relu_ts[4,12] <= 29.32487877957098
 Conv_1_Relu_ts[4,13] <= 3.004389269438119
 Conv_1_Relu_ts[4,14] <= 24.37010483280744
 Conv_1_Relu_ts[4,15] <= 1.234436565832815
 Conv_1_Relu_ts[5,0] <= 25.57325229459289
 Conv_1_Relu_ts[5,1] <= 16.63501010363109
 Conv_1_Relu_ts[5,2] <= 35.00484918336429
 Conv_1_Relu_ts[5,3] <= 0.4998085927964429
 Conv_1_Relu_ts[5,4] <= 57.82216539781905
 Conv_1_Relu_ts[5,5] <= 2.411270213327107
 Conv_1_Relu_ts[5,6] <= 49.46156674751891
 Conv_1_Relu_ts[5,7] <= 3.462243420878994
 Conv_1_Relu_ts[5,8] <= 36.11140265688224
 Conv_1_Relu_ts[5,9] <= 9.030128828439624
 Conv_1_Relu_ts[5,10] <= 23.63937264816158
 Conv_1_Relu_ts[5,11] <= 60.18996818642027
 Conv_1_Relu_ts[5,12] <= 29.32487877957098
 Conv_1_Relu_ts[5,13] <= 3.004389269438119
 Conv_1_Relu_ts[5,14] <= 24.37010483280744
 Conv_1_Relu_ts[5,15] <= 1.234436565832815
 Conv_1_Relu_ts[6,0] <= 25.57325229459289
 Conv_1_Relu_ts[6,1] <= 16.63501010363109
 Conv_1_Relu_ts[6,2] <= 35.00484918336429
 Conv_1_Relu_ts[6,3] <= 0.4998085927964429
 Conv_1_Relu_ts[6,4] <= 57.82216539781905
 Conv_1_Relu_ts[6,5] <= 2.411270213327107
 Conv_1_Relu_ts[6,6] <= 49.46156674751891
 Conv_1_Relu_ts[6,7] <= 3.462243420878994
 Conv_1_Relu_ts[6,8] <= 36.11140265688224
 Conv_1_Relu_ts[6,9] <= 9.030128828439624
 Conv_1_Relu_ts[6,10] <= 23.63937264816158
 Conv_1_Relu_ts[6,11] <= 60.18996818642027
 Conv_1_Relu_ts[6,12] <= 29.32487877957098
 Conv_1_Relu_ts[6,13] <= 3.004389269438119
 Conv_1_Relu_ts[6,14] <= 24.37010483280744
 Conv_1_Relu_ts[6,15] <= 1.234436565832815
 Conv_1_Relu_ts[7,0] <= 25.57325229459289
 Conv_1_Relu_ts[7,1] <= 16.63501010363109
 Conv_1_Relu_ts[7,2] <= 35.00484918336429
 Conv_1_Relu_ts[7,3] <= 0.4998085927964429
 Conv_1_Relu_ts[7,4] <= 57.82216539781905
 Conv_1_Relu_ts[7,5] <= 2.411270213327107
 Conv_1_Relu_ts[7,6] <= 49.46156674751891
 Conv_1_Relu_ts[7,7] <= 3.462243420878994
 Conv_1_Relu_ts[7,8] <= 36.11140265688224
 Conv_1_Relu_ts[7,9] <= 9.030128828439624
 Conv_1_Relu_ts[7,10] <= 23.63937264816158
 Conv_1_Relu_ts[7,11] <= 60.18996818642027
 Conv_1_Relu_ts[7,12] <= 29.32487877957098
 Conv_1_Relu_ts[7,13] <= 3.004389269438119
 Conv_1_Relu_ts[7,14] <= 24.37010483280744
 Conv_1_Relu_ts[7,15] <= 1.234436565832815
 Conv_1_Relu_ts[8,0] <= 25.57325229459289
 Conv_1_Relu_ts[8,1] <= 16.63501010363109
 Conv_1_Relu_ts[8,2] <= 35.00484918336429
 Conv_1_Relu_ts[8,3] <= 0.4998085927964429
 Conv_1_Relu_ts[8,4] <= 57.82216539781905
 Conv_1_Relu_ts[8,5] <= 2.411270213327107
 Conv_1_Relu_ts[8,6] <= 49.46156674751891
 Conv_1_Relu_ts[8,7] <= 3.462243420878994
 Conv_1_Relu_ts[8,8] <= 36.11140265688224
 Conv_1_Relu_ts[8,9] <= 9.030128828439624
 Conv_1_Relu_ts[8,10] <= 23.63937264816158
 Conv_1_Relu_ts[8,11] <= 60.18996818642027
 Conv_1_Relu_ts[8,12] <= 29.32487877957098
 Conv_1_Relu_ts[8,13] <= 3.004389269438119
 Conv_1_Relu_ts[8,14] <= 24.37010483280744
 Conv_1_Relu_ts[8,15] <= 1.234436565832815
 Conv_1_Relu_ts[9,0] <= 25.57325229459289
 Conv_1_Relu_ts[9,1] <= 16.63501010363109
 Conv_1_Relu_ts[9,2] <= 35.00484918336429
 Conv_1_Relu_ts[9,3] <= 0.4998085927964429
 Conv_1_Relu_ts[9,4] <= 57.82216539781905
 Conv_1_Relu_ts[9,5] <= 2.411270213327107
 Conv_1_Relu_ts[9,6] <= 49.46156674751891
 Conv_1_Relu_ts[9,7] <= 3.462243420878994
 Conv_1_Relu_ts[9,8] <= 36.11140265688224
 Conv_1_Relu_ts[9,9] <= 9.030128828439624
 Conv_1_Relu_ts[9,10] <= 23.63937264816158
 Conv_1_Relu_ts[9,11] <= 60.18996818642027
 Conv_1_Relu_ts[9,12] <= 29.32487877957098
 Conv_1_Relu_ts[9,13] <= 3.004389269438119
 Conv_1_Relu_ts[9,14] <= 24.37010483280744
 Conv_1_Relu_ts[9,15] <= 1.234436565832815
 Conv_1_Relu_ss[0,0] <= 56.29245011536772
 Conv_1_Relu_ss[0,1] <= 67.52969714306539
 Conv_1_Relu_ss[0,2] <= 88.95482652838119
 Conv_1_Relu_ss[0,3] <= 33.94596106828702
 Conv_1_Relu_ss[0,4] <= 106.1251358090966
 Conv_1_Relu_ss[0,5] <= 10.77560081932031
 Conv_1_Relu_ss[0,6] <= 29.32218285779035
 Conv_1_Relu_ss[0,7] <= 24.30472028988515
 Conv_1_Relu_ss[0,8] <= 25.8812298511837
 Conv_1_Relu_ss[0,9] <= 76.75665981089216
 Conv_1_Relu_ss[0,10] <= 54.2754182075725
 Conv_1_Relu_ss[0,11] <= 97.18723355279275
 Conv_1_Relu_ss[0,12] <= 47.62093859518428
 Conv_1_Relu_ss[0,13] <= 21.10965662228852
 Conv_1_Relu_ss[0,14] <= 21.61040402775194
 Conv_1_Relu_ss[0,15] <= 11.46170062280965
 Conv_1_Relu_ss[1,0] <= 56.29245011536772
 Conv_1_Relu_ss[1,1] <= 67.52969714306539
 Conv_1_Relu_ss[1,2] <= 88.95482652838119
 Conv_1_Relu_ss[1,3] <= 33.94596106828702
 Conv_1_Relu_ss[1,4] <= 106.1251358090966
 Conv_1_Relu_ss[1,5] <= 10.77560081932031
 Conv_1_Relu_ss[1,6] <= 29.32218285779035
 Conv_1_Relu_ss[1,7] <= 24.30472028988515
 Conv_1_Relu_ss[1,8] <= 25.8812298511837
 Conv_1_Relu_ss[1,9] <= 76.75665981089216
 Conv_1_Relu_ss[1,10] <= 54.2754182075725
 Conv_1_Relu_ss[1,11] <= 97.18723355279275
 Conv_1_Relu_ss[1,12] <= 47.62093859518428
 Conv_1_Relu_ss[1,13] <= 21.10965662228852
 Conv_1_Relu_ss[1,14] <= 21.61040402775194
 Conv_1_Relu_ss[1,15] <= 11.46170062280965
 Conv_1_Relu_ss[2,0] <= 56.29245011536772
 Conv_1_Relu_ss[2,1] <= 67.52969714306539
 Conv_1_Relu_ss[2,2] <= 88.95482652838119
 Conv_1_Relu_ss[2,3] <= 33.94596106828702
 Conv_1_Relu_ss[2,4] <= 106.1251358090966
 Conv_1_Relu_ss[2,5] <= 10.77560081932031
 Conv_1_Relu_ss[2,6] <= 29.32218285779035
 Conv_1_Relu_ss[2,7] <= 24.30472028988515
 Conv_1_Relu_ss[2,8] <= 25.8812298511837
 Conv_1_Relu_ss[2,9] <= 76.75665981089216
 Conv_1_Relu_ss[2,10] <= 54.2754182075725
 Conv_1_Relu_ss[2,11] <= 97.18723355279275
 Conv_1_Relu_ss[2,12] <= 47.62093859518428
 Conv_1_Relu_ss[2,13] <= 21.10965662228852
 Conv_1_Relu_ss[2,14] <= 21.61040402775194
 Conv_1_Relu_ss[2,15] <= 11.46170062280965
 Conv_1_Relu_ss[3,0] <= 56.29245011536772
 Conv_1_Relu_ss[3,1] <= 67.52969714306539
 Conv_1_Relu_ss[3,2] <= 88.95482652838119
 Conv_1_Relu_ss[3,3] <= 33.94596106828702
 Conv_1_Relu_ss[3,4] <= 106.1251358090966
 Conv_1_Relu_ss[3,5] <= 10.77560081932031
 Conv_1_Relu_ss[3,6] <= 29.32218285779035
 Conv_1_Relu_ss[3,7] <= 24.30472028988515
 Conv_1_Relu_ss[3,8] <= 25.8812298511837
 Conv_1_Relu_ss[3,9] <= 76.75665981089216
 Conv_1_Relu_ss[3,10] <= 54.2754182075725
 Conv_1_Relu_ss[3,11] <= 97.18723355279275
 Conv_1_Relu_ss[3,12] <= 47.62093859518428
 Conv_1_Relu_ss[3,13] <= 21.10965662228852
 Conv_1_Relu_ss[3,14] <= 21.61040402775194
 Conv_1_Relu_ss[3,15] <= 11.46170062280965
 Conv_1_Relu_ss[4,0] <= 56.29245011536772
 Conv_1_Relu_ss[4,1] <= 67.52969714306539
 Conv_1_Relu_ss[4,2] <= 88.95482652838119
 Conv_1_Relu_ss[4,3] <= 33.94596106828702
 Conv_1_Relu_ss[4,4] <= 106.1251358090966
 Conv_1_Relu_ss[4,5] <= 10.77560081932031
 Conv_1_Relu_ss[4,6] <= 29.32218285779035
 Conv_1_Relu_ss[4,7] <= 24.30472028988515
 Conv_1_Relu_ss[4,8] <= 25.8812298511837
 Conv_1_Relu_ss[4,9] <= 76.75665981089216
 Conv_1_Relu_ss[4,10] <= 54.2754182075725
 Conv_1_Relu_ss[4,11] <= 97.18723355279275
 Conv_1_Relu_ss[4,12] <= 47.62093859518428
 Conv_1_Relu_ss[4,13] <= 21.10965662228852
 Conv_1_Relu_ss[4,14] <= 21.61040402775194
 Conv_1_Relu_ss[4,15] <= 11.46170062280965
 Conv_1_Relu_ss[5,0] <= 56.29245011536772
 Conv_1_Relu_ss[5,1] <= 67.52969714306539
 Conv_1_Relu_ss[5,2] <= 88.95482652838119
 Conv_1_Relu_ss[5,3] <= 33.94596106828702
 Conv_1_Relu_ss[5,4] <= 106.1251358090966
 Conv_1_Relu_ss[5,5] <= 10.77560081932031
 Conv_1_Relu_ss[5,6] <= 29.32218285779035
 Conv_1_Relu_ss[5,7] <= 24.30472028988515
 Conv_1_Relu_ss[5,8] <= 25.8812298511837
 Conv_1_Relu_ss[5,9] <= 76.75665981089216
 Conv_1_Relu_ss[5,10] <= 54.2754182075725
 Conv_1_Relu_ss[5,11] <= 97.18723355279275
 Conv_1_Relu_ss[5,12] <= 47.62093859518428
 Conv_1_Relu_ss[5,13] <= 21.10965662228852
 Conv_1_Relu_ss[5,14] <= 21.61040402775194
 Conv_1_Relu_ss[5,15] <= 11.46170062280965
 Conv_1_Relu_ss[6,0] <= 56.29245011536772
 Conv_1_Relu_ss[6,1] <= 67.52969714306539
 Conv_1_Relu_ss[6,2] <= 88.95482652838119
 Conv_1_Relu_ss[6,3] <= 33.94596106828702
 Conv_1_Relu_ss[6,4] <= 106.1251358090966
 Conv_1_Relu_ss[6,5] <= 10.77560081932031
 Conv_1_Relu_ss[6,6] <= 29.32218285779035
 Conv_1_Relu_ss[6,7] <= 24.30472028988515
 Conv_1_Relu_ss[6,8] <= 25.8812298511837
 Conv_1_Relu_ss[6,9] <= 76.75665981089216
 Conv_1_Relu_ss[6,10] <= 54.2754182075725
 Conv_1_Relu_ss[6,11] <= 97.18723355279275
 Conv_1_Relu_ss[6,12] <= 47.62093859518428
 Conv_1_Relu_ss[6,13] <= 21.10965662228852
 Conv_1_Relu_ss[6,14] <= 21.61040402775194
 Conv_1_Relu_ss[6,15] <= 11.46170062280965
 Conv_1_Relu_ss[7,0] <= 56.29245011536772
 Conv_1_Relu_ss[7,1] <= 67.52969714306539
 Conv_1_Relu_ss[7,2] <= 88.95482652838119
 Conv_1_Relu_ss[7,3] <= 33.94596106828702
 Conv_1_Relu_ss[7,4] <= 106.1251358090966
 Conv_1_Relu_ss[7,5] <= 10.77560081932031
 Conv_1_Relu_ss[7,6] <= 29.32218285779035
 Conv_1_Relu_ss[7,7] <= 24.30472028988515
 Conv_1_Relu_ss[7,8] <= 25.8812298511837
 Conv_1_Relu_ss[7,9] <= 76.75665981089216
 Conv_1_Relu_ss[7,10] <= 54.2754182075725
 Conv_1_Relu_ss[7,11] <= 97.18723355279275
 Conv_1_Relu_ss[7,12] <= 47.62093859518428
 Conv_1_Relu_ss[7,13] <= 21.10965662228852
 Conv_1_Relu_ss[7,14] <= 21.61040402775194
 Conv_1_Relu_ss[7,15] <= 11.46170062280965
 Conv_1_Relu_ss[8,0] <= 56.29245011536772
 Conv_1_Relu_ss[8,1] <= 67.52969714306539
 Conv_1_Relu_ss[8,2] <= 88.95482652838119
 Conv_1_Relu_ss[8,3] <= 33.94596106828702
 Conv_1_Relu_ss[8,4] <= 106.1251358090966
 Conv_1_Relu_ss[8,5] <= 10.77560081932031
 Conv_1_Relu_ss[8,6] <= 29.32218285779035
 Conv_1_Relu_ss[8,7] <= 24.30472028988515
 Conv_1_Relu_ss[8,8] <= 25.8812298511837
 Conv_1_Relu_ss[8,9] <= 76.75665981089216
 Conv_1_Relu_ss[8,10] <= 54.2754182075725
 Conv_1_Relu_ss[8,11] <= 97.18723355279275
 Conv_1_Relu_ss[8,12] <= 47.62093859518428
 Conv_1_Relu_ss[8,13] <= 21.10965662228852
 Conv_1_Relu_ss[8,14] <= 21.61040402775194
 Conv_1_Relu_ss[8,15] <= 11.46170062280965
 Conv_1_Relu_ss[9,0] <= 56.29245011536772
 Conv_1_Relu_ss[9,1] <= 67.52969714306539
 Conv_1_Relu_ss[9,2] <= 88.95482652838119
 Conv_1_Relu_ss[9,3] <= 33.94596106828702
 Conv_1_Relu_ss[9,4] <= 106.1251358090966
 Conv_1_Relu_ss[9,5] <= 10.77560081932031
 Conv_1_Relu_ss[9,6] <= 29.32218285779035
 Conv_1_Relu_ss[9,7] <= 24.30472028988515
 Conv_1_Relu_ss[9,8] <= 25.8812298511837
 Conv_1_Relu_ss[9,9] <= 76.75665981089216
 Conv_1_Relu_ss[9,10] <= 54.2754182075725
 Conv_1_Relu_ss[9,11] <= 97.18723355279275
 Conv_1_Relu_ss[9,12] <= 47.62093859518428
 Conv_1_Relu_ss[9,13] <= 21.10965662228852
 Conv_1_Relu_ss[9,14] <= 21.61040402775194
 Conv_1_Relu_ss[9,15] <= 11.46170062280965
 Conv_2_aggregated_features[0,0] <= 25.57325229459289
 Conv_2_aggregated_features[0,1] <= 16.63501010363109
 Conv_2_aggregated_features[0,2] <= 35.00484918336429
 Conv_2_aggregated_features[0,3] <= 0.4998085927964429
 Conv_2_aggregated_features[0,4] <= 57.82216539781905
 Conv_2_aggregated_features[0,5] <= 2.411270213327107
 Conv_2_aggregated_features[0,6] <= 49.46156674751892
 Conv_2_aggregated_features[0,7] <= 3.462243420878994
 Conv_2_aggregated_features[0,8] <= 36.11140265688225
 Conv_2_aggregated_features[0,9] <= 9.030128828439626
 Conv_2_aggregated_features[0,10] <= 23.63937264816158
 Conv_2_aggregated_features[0,11] <= 60.18996818642027
 Conv_2_aggregated_features[0,12] <= 29.32487877957099
 Conv_2_aggregated_features[0,13] <= 3.004389269438118
 Conv_2_aggregated_features[0,14] <= 24.37010483280744
 Conv_2_aggregated_features[0,15] <= 1.234436565832815
 Conv_2_aggregated_features[1,0] <= 25.57325229459289
 Conv_2_aggregated_features[1,1] <= 16.63501010363109
 Conv_2_aggregated_features[1,2] <= 35.00484918336429
 Conv_2_aggregated_features[1,3] <= 0.4998085927964429
 Conv_2_aggregated_features[1,4] <= 57.82216539781905
 Conv_2_aggregated_features[1,5] <= 2.411270213327107
 Conv_2_aggregated_features[1,6] <= 49.46156674751892
 Conv_2_aggregated_features[1,7] <= 3.462243420878994
 Conv_2_aggregated_features[1,8] <= 36.11140265688225
 Conv_2_aggregated_features[1,9] <= 9.030128828439626
 Conv_2_aggregated_features[1,10] <= 23.63937264816158
 Conv_2_aggregated_features[1,11] <= 60.18996818642027
 Conv_2_aggregated_features[1,12] <= 29.32487877957099
 Conv_2_aggregated_features[1,13] <= 3.004389269438118
 Conv_2_aggregated_features[1,14] <= 24.37010483280744
 Conv_2_aggregated_features[1,15] <= 1.234436565832815
 Conv_2_aggregated_features[2,0] <= 25.57325229459289
 Conv_2_aggregated_features[2,1] <= 16.63501010363109
 Conv_2_aggregated_features[2,2] <= 35.00484918336429
 Conv_2_aggregated_features[2,3] <= 0.4998085927964429
 Conv_2_aggregated_features[2,4] <= 57.82216539781905
 Conv_2_aggregated_features[2,5] <= 2.411270213327107
 Conv_2_aggregated_features[2,6] <= 49.46156674751892
 Conv_2_aggregated_features[2,7] <= 3.462243420878994
 Conv_2_aggregated_features[2,8] <= 36.11140265688225
 Conv_2_aggregated_features[2,9] <= 9.030128828439626
 Conv_2_aggregated_features[2,10] <= 23.63937264816158
 Conv_2_aggregated_features[2,11] <= 60.18996818642027
 Conv_2_aggregated_features[2,12] <= 29.32487877957099
 Conv_2_aggregated_features[2,13] <= 3.004389269438118
 Conv_2_aggregated_features[2,14] <= 24.37010483280744
 Conv_2_aggregated_features[2,15] <= 1.234436565832815
 Conv_2_aggregated_features[3,0] <= 25.57325229459289
 Conv_2_aggregated_features[3,1] <= 16.63501010363109
 Conv_2_aggregated_features[3,2] <= 35.00484918336429
 Conv_2_aggregated_features[3,3] <= 0.4998085927964429
 Conv_2_aggregated_features[3,4] <= 57.82216539781905
 Conv_2_aggregated_features[3,5] <= 2.411270213327107
 Conv_2_aggregated_features[3,6] <= 49.46156674751892
 Conv_2_aggregated_features[3,7] <= 3.462243420878994
 Conv_2_aggregated_features[3,8] <= 36.11140265688225
 Conv_2_aggregated_features[3,9] <= 9.030128828439626
 Conv_2_aggregated_features[3,10] <= 23.63937264816158
 Conv_2_aggregated_features[3,11] <= 60.18996818642027
 Conv_2_aggregated_features[3,12] <= 29.32487877957099
 Conv_2_aggregated_features[3,13] <= 3.004389269438118
 Conv_2_aggregated_features[3,14] <= 24.37010483280744
 Conv_2_aggregated_features[3,15] <= 1.234436565832815
 Conv_2_aggregated_features[4,0] <= 25.57325229459289
 Conv_2_aggregated_features[4,1] <= 16.63501010363109
 Conv_2_aggregated_features[4,2] <= 35.00484918336429
 Conv_2_aggregated_features[4,3] <= 0.4998085927964429
 Conv_2_aggregated_features[4,4] <= 57.82216539781905
 Conv_2_aggregated_features[4,5] <= 2.411270213327107
 Conv_2_aggregated_features[4,6] <= 49.46156674751892
 Conv_2_aggregated_features[4,7] <= 3.462243420878994
 Conv_2_aggregated_features[4,8] <= 36.11140265688225
 Conv_2_aggregated_features[4,9] <= 9.030128828439626
 Conv_2_aggregated_features[4,10] <= 23.63937264816158
 Conv_2_aggregated_features[4,11] <= 60.18996818642027
 Conv_2_aggregated_features[4,12] <= 29.32487877957099
 Conv_2_aggregated_features[4,13] <= 3.004389269438118
 Conv_2_aggregated_features[4,14] <= 24.37010483280744
 Conv_2_aggregated_features[4,15] <= 1.234436565832815
 Conv_2_aggregated_features[5,0] <= 25.57325229459289
 Conv_2_aggregated_features[5,1] <= 16.63501010363109
 Conv_2_aggregated_features[5,2] <= 35.00484918336429
 Conv_2_aggregated_features[5,3] <= 0.4998085927964429
 Conv_2_aggregated_features[5,4] <= 57.82216539781905
 Conv_2_aggregated_features[5,5] <= 2.411270213327107
 Conv_2_aggregated_features[5,6] <= 49.46156674751892
 Conv_2_aggregated_features[5,7] <= 3.462243420878994
 Conv_2_aggregated_features[5,8] <= 36.11140265688225
 Conv_2_aggregated_features[5,9] <= 9.030128828439626
 Conv_2_aggregated_features[5,10] <= 23.63937264816158
 Conv_2_aggregated_features[5,11] <= 60.18996818642027
 Conv_2_aggregated_features[5,12] <= 29.32487877957099
 Conv_2_aggregated_features[5,13] <= 3.004389269438118
 Conv_2_aggregated_features[5,14] <= 24.37010483280744
 Conv_2_aggregated_features[5,15] <= 1.234436565832815
 Conv_2_aggregated_features[6,0] <= 25.57325229459289
 Conv_2_aggregated_features[6,1] <= 16.63501010363109
 Conv_2_aggregated_features[6,2] <= 35.00484918336429
 Conv_2_aggregated_features[6,3] <= 0.4998085927964429
 Conv_2_aggregated_features[6,4] <= 57.82216539781905
 Conv_2_aggregated_features[6,5] <= 2.411270213327107
 Conv_2_aggregated_features[6,6] <= 49.46156674751892
 Conv_2_aggregated_features[6,7] <= 3.462243420878994
 Conv_2_aggregated_features[6,8] <= 36.11140265688225
 Conv_2_aggregated_features[6,9] <= 9.030128828439626
 Conv_2_aggregated_features[6,10] <= 23.63937264816158
 Conv_2_aggregated_features[6,11] <= 60.18996818642027
 Conv_2_aggregated_features[6,12] <= 29.32487877957099
 Conv_2_aggregated_features[6,13] <= 3.004389269438118
 Conv_2_aggregated_features[6,14] <= 24.37010483280744
 Conv_2_aggregated_features[6,15] <= 1.234436565832815
 Conv_2_aggregated_features[7,0] <= 25.57325229459289
 Conv_2_aggregated_features[7,1] <= 16.63501010363109
 Conv_2_aggregated_features[7,2] <= 35.00484918336429
 Conv_2_aggregated_features[7,3] <= 0.4998085927964429
 Conv_2_aggregated_features[7,4] <= 57.82216539781905
 Conv_2_aggregated_features[7,5] <= 2.411270213327107
 Conv_2_aggregated_features[7,6] <= 49.46156674751892
 Conv_2_aggregated_features[7,7] <= 3.462243420878994
 Conv_2_aggregated_features[7,8] <= 36.11140265688225
 Conv_2_aggregated_features[7,9] <= 9.030128828439626
 Conv_2_aggregated_features[7,10] <= 23.63937264816158
 Conv_2_aggregated_features[7,11] <= 60.18996818642027
 Conv_2_aggregated_features[7,12] <= 29.32487877957099
 Conv_2_aggregated_features[7,13] <= 3.004389269438118
 Conv_2_aggregated_features[7,14] <= 24.37010483280744
 Conv_2_aggregated_features[7,15] <= 1.234436565832815
 Conv_2_aggregated_features[8,0] <= 25.57325229459289
 Conv_2_aggregated_features[8,1] <= 16.63501010363109
 Conv_2_aggregated_features[8,2] <= 35.00484918336429
 Conv_2_aggregated_features[8,3] <= 0.4998085927964429
 Conv_2_aggregated_features[8,4] <= 57.82216539781905
 Conv_2_aggregated_features[8,5] <= 2.411270213327107
 Conv_2_aggregated_features[8,6] <= 49.46156674751892
 Conv_2_aggregated_features[8,7] <= 3.462243420878994
 Conv_2_aggregated_features[8,8] <= 36.11140265688225
 Conv_2_aggregated_features[8,9] <= 9.030128828439626
 Conv_2_aggregated_features[8,10] <= 23.63937264816158
 Conv_2_aggregated_features[8,11] <= 60.18996818642027
 Conv_2_aggregated_features[8,12] <= 29.32487877957099
 Conv_2_aggregated_features[8,13] <= 3.004389269438118
 Conv_2_aggregated_features[8,14] <= 24.37010483280744
 Conv_2_aggregated_features[8,15] <= 1.234436565832815
 Conv_2_aggregated_features[9,0] <= 25.57325229459289
 Conv_2_aggregated_features[9,1] <= 16.63501010363109
 Conv_2_aggregated_features[9,2] <= 35.00484918336429
 Conv_2_aggregated_features[9,3] <= 0.4998085927964429
 Conv_2_aggregated_features[9,4] <= 57.82216539781905
 Conv_2_aggregated_features[9,5] <= 2.411270213327107
 Conv_2_aggregated_features[9,6] <= 49.46156674751892
 Conv_2_aggregated_features[9,7] <= 3.462243420878994
 Conv_2_aggregated_features[9,8] <= 36.11140265688225
 Conv_2_aggregated_features[9,9] <= 9.030128828439626
 Conv_2_aggregated_features[9,10] <= 23.63937264816158
 Conv_2_aggregated_features[9,11] <= 60.18996818642027
 Conv_2_aggregated_features[9,12] <= 29.32487877957099
 Conv_2_aggregated_features[9,13] <= 3.004389269438118
 Conv_2_aggregated_features[9,14] <= 24.37010483280744
 Conv_2_aggregated_features[9,15] <= 1.234436565832815
 -256 <= Conv_2_t[0,0] <= 256
 -256 <= Conv_2_t[0,1] <= 126.8091092372764
 -256 <= Conv_2_t[0,2] <= 252.2492312397784
 -256 <= Conv_2_t[0,3] <= 256
 -242.8364949340469 <= Conv_2_t[0,4] <= 256
 -256 <= Conv_2_t[0,5] <= 256
 -96.48911034049161 <= Conv_2_t[0,6] <= 256
 -101.2542460305265 <= Conv_2_t[0,7] <= 11.47895694870232
 -256 <= Conv_2_t[1,0] <= 256
 -256 <= Conv_2_t[1,1] <= 126.8091092372764
 -256 <= Conv_2_t[1,2] <= 252.2492312397784
 -256 <= Conv_2_t[1,3] <= 256
 -242.8364949340469 <= Conv_2_t[1,4] <= 256
 -256 <= Conv_2_t[1,5] <= 256
 -96.48911034049161 <= Conv_2_t[1,6] <= 256
 -101.2542460305265 <= Conv_2_t[1,7] <= 11.47895694870232
 -256 <= Conv_2_t[2,0] <= 256
 -256 <= Conv_2_t[2,1] <= 126.8091092372764
 -256 <= Conv_2_t[2,2] <= 252.2492312397784
 -256 <= Conv_2_t[2,3] <= 256
 -242.8364949340469 <= Conv_2_t[2,4] <= 256
 -256 <= Conv_2_t[2,5] <= 256
 -96.48911034049161 <= Conv_2_t[2,6] <= 256
 -101.2542460305265 <= Conv_2_t[2,7] <= 11.47895694870232
 -256 <= Conv_2_t[3,0] <= 256
 -256 <= Conv_2_t[3,1] <= 126.8091092372764
 -256 <= Conv_2_t[3,2] <= 252.2492312397784
 -256 <= Conv_2_t[3,3] <= 256
 -242.8364949340469 <= Conv_2_t[3,4] <= 256
 -256 <= Conv_2_t[3,5] <= 256
 -96.48911034049161 <= Conv_2_t[3,6] <= 256
 -101.2542460305265 <= Conv_2_t[3,7] <= 11.47895694870232
 -256 <= Conv_2_t[4,0] <= 256
 -256 <= Conv_2_t[4,1] <= 126.8091092372764
 -256 <= Conv_2_t[4,2] <= 252.2492312397784
 -256 <= Conv_2_t[4,3] <= 256
 -242.8364949340469 <= Conv_2_t[4,4] <= 256
 -256 <= Conv_2_t[4,5] <= 256
 -96.48911034049161 <= Conv_2_t[4,6] <= 256
 -101.2542460305265 <= Conv_2_t[4,7] <= 11.47895694870232
 -256 <= Conv_2_t[5,0] <= 256
 -256 <= Conv_2_t[5,1] <= 126.8091092372764
 -256 <= Conv_2_t[5,2] <= 252.2492312397784
 -256 <= Conv_2_t[5,3] <= 256
 -242.8364949340469 <= Conv_2_t[5,4] <= 256
 -256 <= Conv_2_t[5,5] <= 256
 -96.48911034049161 <= Conv_2_t[5,6] <= 256
 -101.2542460305265 <= Conv_2_t[5,7] <= 11.47895694870232
 -256 <= Conv_2_t[6,0] <= 256
 -256 <= Conv_2_t[6,1] <= 126.8091092372764
 -256 <= Conv_2_t[6,2] <= 252.2492312397784
 -256 <= Conv_2_t[6,3] <= 256
 -242.8364949340469 <= Conv_2_t[6,4] <= 256
 -256 <= Conv_2_t[6,5] <= 256
 -96.48911034049161 <= Conv_2_t[6,6] <= 256
 -101.2542460305265 <= Conv_2_t[6,7] <= 11.47895694870232
 -256 <= Conv_2_t[7,0] <= 256
 -256 <= Conv_2_t[7,1] <= 126.8091092372764
 -256 <= Conv_2_t[7,2] <= 252.2492312397784
 -256 <= Conv_2_t[7,3] <= 256
 -242.8364949340469 <= Conv_2_t[7,4] <= 256
 -256 <= Conv_2_t[7,5] <= 256
 -96.48911034049161 <= Conv_2_t[7,6] <= 256
 -101.2542460305265 <= Conv_2_t[7,7] <= 11.47895694870232
 -256 <= Conv_2_t[8,0] <= 256
 -256 <= Conv_2_t[8,1] <= 126.8091092372764
 -256 <= Conv_2_t[8,2] <= 252.2492312397784
 -256 <= Conv_2_t[8,3] <= 256
 -242.8364949340469 <= Conv_2_t[8,4] <= 256
 -256 <= Conv_2_t[8,5] <= 256
 -96.48911034049161 <= Conv_2_t[8,6] <= 256
 -101.2542460305265 <= Conv_2_t[8,7] <= 11.47895694870232
 -256 <= Conv_2_t[9,0] <= 256
 -256 <= Conv_2_t[9,1] <= 126.8091092372764
 -256 <= Conv_2_t[9,2] <= 252.2492312397784
 -256 <= Conv_2_t[9,3] <= 256
 -242.8364949340469 <= Conv_2_t[9,4] <= 256
 -256 <= Conv_2_t[9,5] <= 256
 -96.48911034049161 <= Conv_2_t[9,6] <= 256
 -101.2542460305265 <= Conv_2_t[9,7] <= 11.47895694870232
 Conv_2_Relu_ts[0,0] <= 256
 Conv_2_Relu_ts[0,1] <= 126.8091092372764
 Conv_2_Relu_ts[0,2] <= 252.2492312397784
 Conv_2_Relu_ts[0,3] <= 256
 Conv_2_Relu_ts[0,4] <= 256
 Conv_2_Relu_ts[0,5] <= 256
 Conv_2_Relu_ts[0,6] <= 256
 Conv_2_Relu_ts[0,7] <= 11.47895694870232
 Conv_2_Relu_ts[1,0] <= 256
 Conv_2_Relu_ts[1,1] <= 126.8091092372764
 Conv_2_Relu_ts[1,2] <= 252.2492312397784
 Conv_2_Relu_ts[1,3] <= 256
 Conv_2_Relu_ts[1,4] <= 256
 Conv_2_Relu_ts[1,5] <= 256
 Conv_2_Relu_ts[1,6] <= 256
 Conv_2_Relu_ts[1,7] <= 11.47895694870232
 Conv_2_Relu_ts[2,0] <= 256
 Conv_2_Relu_ts[2,1] <= 126.8091092372764
 Conv_2_Relu_ts[2,2] <= 252.2492312397784
 Conv_2_Relu_ts[2,3] <= 256
 Conv_2_Relu_ts[2,4] <= 256
 Conv_2_Relu_ts[2,5] <= 256
 Conv_2_Relu_ts[2,6] <= 256
 Conv_2_Relu_ts[2,7] <= 11.47895694870232
 Conv_2_Relu_ts[3,0] <= 256
 Conv_2_Relu_ts[3,1] <= 126.8091092372764
 Conv_2_Relu_ts[3,2] <= 252.2492312397784
 Conv_2_Relu_ts[3,3] <= 256
 Conv_2_Relu_ts[3,4] <= 256
 Conv_2_Relu_ts[3,5] <= 256
 Conv_2_Relu_ts[3,6] <= 256
 Conv_2_Relu_ts[3,7] <= 11.47895694870232
 Conv_2_Relu_ts[4,0] <= 256
 Conv_2_Relu_ts[4,1] <= 126.8091092372764
 Conv_2_Relu_ts[4,2] <= 252.2492312397784
 Conv_2_Relu_ts[4,3] <= 256
 Conv_2_Relu_ts[4,4] <= 256
 Conv_2_Relu_ts[4,5] <= 256
 Conv_2_Relu_ts[4,6] <= 256
 Conv_2_Relu_ts[4,7] <= 11.47895694870232
 Conv_2_Relu_ts[5,0] <= 256
 Conv_2_Relu_ts[5,1] <= 126.8091092372764
 Conv_2_Relu_ts[5,2] <= 252.2492312397784
 Conv_2_Relu_ts[5,3] <= 256
 Conv_2_Relu_ts[5,4] <= 256
 Conv_2_Relu_ts[5,5] <= 256
 Conv_2_Relu_ts[5,6] <= 256
 Conv_2_Relu_ts[5,7] <= 11.47895694870232
 Conv_2_Relu_ts[6,0] <= 256
 Conv_2_Relu_ts[6,1] <= 126.8091092372764
 Conv_2_Relu_ts[6,2] <= 252.2492312397784
 Conv_2_Relu_ts[6,3] <= 256
 Conv_2_Relu_ts[6,4] <= 256
 Conv_2_Relu_ts[6,5] <= 256
 Conv_2_Relu_ts[6,6] <= 256
 Conv_2_Relu_ts[6,7] <= 11.47895694870232
 Conv_2_Relu_ts[7,0] <= 256
 Conv_2_Relu_ts[7,1] <= 126.8091092372764
 Conv_2_Relu_ts[7,2] <= 252.2492312397784
 Conv_2_Relu_ts[7,3] <= 256
 Conv_2_Relu_ts[7,4] <= 256
 Conv_2_Relu_ts[7,5] <= 256
 Conv_2_Relu_ts[7,6] <= 256
 Conv_2_Relu_ts[7,7] <= 11.47895694870232
 Conv_2_Relu_ts[8,0] <= 256
 Conv_2_Relu_ts[8,1] <= 126.8091092372764
 Conv_2_Relu_ts[8,2] <= 252.2492312397784
 Conv_2_Relu_ts[8,3] <= 256
 Conv_2_Relu_ts[8,4] <= 256
 Conv_2_Relu_ts[8,5] <= 256
 Conv_2_Relu_ts[8,6] <= 256
 Conv_2_Relu_ts[8,7] <= 11.47895694870232
 Conv_2_Relu_ts[9,0] <= 256
 Conv_2_Relu_ts[9,1] <= 126.8091092372764
 Conv_2_Relu_ts[9,2] <= 252.2492312397784
 Conv_2_Relu_ts[9,3] <= 256
 Conv_2_Relu_ts[9,4] <= 256
 Conv_2_Relu_ts[9,5] <= 256
 Conv_2_Relu_ts[9,6] <= 256
 Conv_2_Relu_ts[9,7] <= 11.47895694870232
 Conv_2_Relu_ss[0,0] <= 256
 Conv_2_Relu_ss[0,1] <= 256
 Conv_2_Relu_ss[0,2] <= 256
 Conv_2_Relu_ss[0,3] <= 256
 Conv_2_Relu_ss[0,4] <= 242.8364949340469
 Conv_2_Relu_ss[0,5] <= 256
 Conv_2_Relu_ss[0,6] <= 96.48911034049161
 Conv_2_Relu_ss[0,7] <= 101.2542460305265
 Conv_2_Relu_ss[1,0] <= 256
 Conv_2_Relu_ss[1,1] <= 256
 Conv_2_Relu_ss[1,2] <= 256
 Conv_2_Relu_ss[1,3] <= 256
 Conv_2_Relu_ss[1,4] <= 242.8364949340469
 Conv_2_Relu_ss[1,5] <= 256
 Conv_2_Relu_ss[1,6] <= 96.48911034049161
 Conv_2_Relu_ss[1,7] <= 101.2542460305265
 Conv_2_Relu_ss[2,0] <= 256
 Conv_2_Relu_ss[2,1] <= 256
 Conv_2_Relu_ss[2,2] <= 256
 Conv_2_Relu_ss[2,3] <= 256
 Conv_2_Relu_ss[2,4] <= 242.8364949340469
 Conv_2_Relu_ss[2,5] <= 256
 Conv_2_Relu_ss[2,6] <= 96.48911034049161
 Conv_2_Relu_ss[2,7] <= 101.2542460305265
 Conv_2_Relu_ss[3,0] <= 256
 Conv_2_Relu_ss[3,1] <= 256
 Conv_2_Relu_ss[3,2] <= 256
 Conv_2_Relu_ss[3,3] <= 256
 Conv_2_Relu_ss[3,4] <= 242.8364949340469
 Conv_2_Relu_ss[3,5] <= 256
 Conv_2_Relu_ss[3,6] <= 96.48911034049161
 Conv_2_Relu_ss[3,7] <= 101.2542460305265
 Conv_2_Relu_ss[4,0] <= 256
 Conv_2_Relu_ss[4,1] <= 256
 Conv_2_Relu_ss[4,2] <= 256
 Conv_2_Relu_ss[4,3] <= 256
 Conv_2_Relu_ss[4,4] <= 242.8364949340469
 Conv_2_Relu_ss[4,5] <= 256
 Conv_2_Relu_ss[4,6] <= 96.48911034049161
 Conv_2_Relu_ss[4,7] <= 101.2542460305265
 Conv_2_Relu_ss[5,0] <= 256
 Conv_2_Relu_ss[5,1] <= 256
 Conv_2_Relu_ss[5,2] <= 256
 Conv_2_Relu_ss[5,3] <= 256
 Conv_2_Relu_ss[5,4] <= 242.8364949340469
 Conv_2_Relu_ss[5,5] <= 256
 Conv_2_Relu_ss[5,6] <= 96.48911034049161
 Conv_2_Relu_ss[5,7] <= 101.2542460305265
 Conv_2_Relu_ss[6,0] <= 256
 Conv_2_Relu_ss[6,1] <= 256
 Conv_2_Relu_ss[6,2] <= 256
 Conv_2_Relu_ss[6,3] <= 256
 Conv_2_Relu_ss[6,4] <= 242.8364949340469
 Conv_2_Relu_ss[6,5] <= 256
 Conv_2_Relu_ss[6,6] <= 96.48911034049161
 Conv_2_Relu_ss[6,7] <= 101.2542460305265
 Conv_2_Relu_ss[7,0] <= 256
 Conv_2_Relu_ss[7,1] <= 256
 Conv_2_Relu_ss[7,2] <= 256
 Conv_2_Relu_ss[7,3] <= 256
 Conv_2_Relu_ss[7,4] <= 242.8364949340469
 Conv_2_Relu_ss[7,5] <= 256
 Conv_2_Relu_ss[7,6] <= 96.48911034049161
 Conv_2_Relu_ss[7,7] <= 101.2542460305265
 Conv_2_Relu_ss[8,0] <= 256
 Conv_2_Relu_ss[8,1] <= 256
 Conv_2_Relu_ss[8,2] <= 256
 Conv_2_Relu_ss[8,3] <= 256
 Conv_2_Relu_ss[8,4] <= 242.8364949340469
 Conv_2_Relu_ss[8,5] <= 256
 Conv_2_Relu_ss[8,6] <= 96.48911034049161
 Conv_2_Relu_ss[8,7] <= 101.2542460305265
 Conv_2_Relu_ss[9,0] <= 256
 Conv_2_Relu_ss[9,1] <= 256
 Conv_2_Relu_ss[9,2] <= 256
 Conv_2_Relu_ss[9,3] <= 256
 Conv_2_Relu_ss[9,4] <= 242.8364949340469
 Conv_2_Relu_ss[9,5] <= 256
 Conv_2_Relu_ss[9,6] <= 96.48911034049161
 Conv_2_Relu_ss[9,7] <= 101.2542460305265
 Aggregation[0] <= 256
 Aggregation[1] <= 126.8091092372764
 Aggregation[2] <= 252.2492312397785
 Aggregation[3] <= 256
 Aggregation[4] <= 256
 Aggregation[5] <= 256
 Aggregation[6] <= 256
 Aggregation[7] <= 11.47895694870232
 -256 <= Lin_0_t[0] <= 256
 -256 <= Lin_0_t[1] <= 256
 -256 <= Lin_0_t[2] <= 256
 -256 <= Lin_0_t[3] <= 15.74425860767926
 -256 <= Lin_0_t[4] <= 256
 -114.0982224191812 <= Lin_0_t[5] <= 256
 -256 <= Lin_0_t[6] <= 256
 -256 <= Lin_0_t[7] <= 232.4072593565022
 Lin_0_Relu_ts[0,0] <= 256
 Lin_0_Relu_ts[0,1] <= 256
 Lin_0_Relu_ts[0,2] <= 256
 Lin_0_Relu_ts[0,3] <= 15.74425860767926
 Lin_0_Relu_ts[0,4] <= 256
 Lin_0_Relu_ts[0,5] <= 256
 Lin_0_Relu_ts[0,6] <= 256
 Lin_0_Relu_ts[0,7] <= 232.4072593565022
 Lin_0_Relu_ss[0,0] <= 256
 Lin_0_Relu_ss[0,1] <= 256
 Lin_0_Relu_ss[0,2] <= 256
 Lin_0_Relu_ss[0,3] <= 256
 Lin_0_Relu_ss[0,4] <= 256
 Lin_0_Relu_ss[0,5] <= 114.0982224191812
 Lin_0_Relu_ss[0,6] <= 256
 Lin_0_Relu_ss[0,7] <= 256
 -256 <= Lin_1_t[0] <= 256
 -256 <= Lin_1_t[1] <= 116.6207733154297
 -256 <= Lin_1_t[2] <= 256
 -256 <= Lin_1_t[3] <= 256
 -256 <= Lin_1_t[4] <= 1.539772889596566
 -256 <= Lin_1_t[5] <= 256
 -256 <= Lin_1_t[6] <= 98.27742641179015
 -256 <= Lin_1_t[7] <= 256
 Lin1Relu_ts[0,0] <= 256
 Lin1Relu_ts[0,1] <= 116.6207733154297
 Lin1Relu_ts[0,2] <= 256
 Lin1Relu_ts[0,3] <= 256
 Lin1Relu_ts[0,4] <= 1.539772889596566
 Lin1Relu_ts[0,5] <= 256
 Lin1Relu_ts[0,6] <= 98.27742641179015
 Lin1Relu_ts[0,7] <= 256
 Lin1Relu_ss[0,0] <= 256
 Lin1Relu_ss[0,1] <= 256
 Lin1Relu_ss[0,2] <= 256
 Lin1Relu_ss[0,3] <= 256
 Lin1Relu_ss[0,4] <= 256
 Lin1Relu_ss[0,5] <= 256
 Lin1Relu_ss[0,6] <= 256
 Lin1Relu_ss[0,7] <= 256
 -256 <= Output_t[0] <= 256
 -256 <= Output_t[1] <= 256
 -256 <= Output_t[2] <= 256
 -256 <= Output_t[3] <= 256
 -256 <= Output_t[4] <= 217.0288423316715
 embedding_similarity <= 1
Binaries
 A[0,0] A[0,1] A[0,2] A[0,3] A[0,4] A[0,5] A[0,6] A[0,7] A[0,8] A[0,9]
 A[1,0] A[1,1] A[1,2] A[1,3] A[1,4] A[1,5] A[1,6] A[1,7] A[1,8] A[1,9]
 A[2,0] A[2,1] A[2,2] A[2,3] A[2,4] A[2,5] A[2,6] A[2,7] A[2,8] A[2,9]
 A[3,0] A[3,1] A[3,2] A[3,3] A[3,4] A[3,5] A[3,6] A[3,7] A[3,8] A[3,9]
 A[4,0] A[4,1] A[4,2] A[4,3] A[4,4] A[4,5] A[4,6] A[4,7] A[4,8] A[4,9]
 A[5,0] A[5,1] A[5,2] A[5,3] A[5,4] A[5,5] A[5,6] A[5,7] A[5,8] A[5,9]
 A[6,0] A[6,1] A[6,2] A[6,3] A[6,4] A[6,5] A[6,6] A[6,7] A[6,8] A[6,9]
 A[7,0] A[7,1] A[7,2] A[7,3] A[7,4] A[7,5] A[7,6] A[7,7] A[7,8] A[7,9]
 A[8,0] A[8,1] A[8,2] A[8,3] A[8,4] A[8,5] A[8,6] A[8,7] A[8,8] A[8,9]
 A[9,0] A[9,1] A[9,2] A[9,3] A[9,4] A[9,5] A[9,6] A[9,7] A[9,8] A[9,9] C600
 C601 C602 C603 C604 C605 C606 C607 C608 C609 C610 C611 C612 C613 C614 C615
 C616 C617 C618 C619 C620 C621 C622 C623 C624 C625 C626 C627 C628 C629 C630
 C631 C632 C633 C634 C635 C636 C637 C638 C639 C640 C641 C642 C643 C644 C645
 C646 C647 C648 C649 C650 C651 C652 C653 C654 C655 C656 C657 C658 C659 C660
 C661 C662 C663 C664 C665 C666 C667 C668 C669 C670 C671 C672 C673 C674 C675
 C676 C677 C678 C679 C680 C681 C682 C683 C684 C685 C686 C687 C688 C689 C690
 C691 C692 C693 C694 C695 C696 C697 C698 C699 C700 C701 C702 C703 C704 C705
 C706 C707 C708 C709 C710 C711 C712 C713 C714 C715 C716 C717 C718 C719 C720
 C721 C722 C723 C724 C725 C726 C727 C728 C729 C730 C731 C732 C733 C734 C735
 C736 C737 C738 C739 C740 C741 C742 C743 C744 C745 C746 C747 C748 C749 C750
 C751 C752 C753 C754 C755 C756 C757 C758 C759 C1400 C1401 C1402 C1403 C1404
 C1405 C1406 C1407 C1408 C1409 C1410 C1411 C1412 C1413 C1414 C1415 C1416
 C1417 C1418 C1419 C1420 C1421 C1422 C1423 C1424 C1425 C1426 C1427 C1428
 C1429 C1430 C1431 C1432 C1433 C1434 C1435 C1436 C1437 C1438 C1439 C1440
 C1441 C1442 C1443 C1444 C1445 C1446 C1447 C1448 C1449 C1450 C1451 C1452
 C1453 C1454 C1455 C1456 C1457 C1458 C1459 C1460 C1461 C1462 C1463 C1464
 C1465 C1466 C1467 C1468 C1469 C1470 C1471 C1472 C1473 C1474 C1475 C1476
 C1477 C1478 C1479 C1480 C1481 C1482 C1483 C1484 C1485 C1486 C1487 C1488
 C1489 C1490 C1491 C1492 C1493 C1494 C1495 C1496 C1497 C1498 C1499 C1500
 C1501 C1502 C1503 C1504 C1505 C1506 C1507 C1508 C1509 C1510 C1511 C1512
 C1513 C1514 C1515 C1516 C1517 C1518 C1519 C1520 C1521 C1522 C1523 C1524
 C1525 C1526 C1527 C1528 C1529 C1530 C1531 C1532 C1533 C1534 C1535 C1536
 C1537 C1538 C1539 C1540 C1541 C1542 C1543 C1544 C1545 C1546 C1547 C1548
 C1549 C1550 C1551 C1552 C1553 C1554 C1555 C1556 C1557 C1558 C1559 C1960
 C1961 C1962 C1963 C1964 C1965 C1966 C1967 C1968 C1969 C1970 C1971 C1972
 C1973 C1974 C1975 C1976 C1977 C1978 C1979 C1980 C1981 C1982 C1983 C1984
 C1985 C1986 C1987 C1988 C1989 C1990 C1991 C1992 C1993 C1994 C1995 C1996
 C1997 C1998 C1999 C2000 C2001 C2002 C2003 C2004 C2005 C2006 C2007 C2008
 C2009 C2010 C2011 C2012 C2013 C2014 C2015 C2016 C2017 C2018 C2019 C2020
 C2021 C2022 C2023 C2024 C2025 C2026 C2027 C2028 C2029 C2030 C2031 C2032
 C2033 C2034 C2035 C2036 C2037 C2038 C2039 C2072 C2073 C2074 C2075 C2076
 C2077 C2078 C2079 C2104 C2105 C2106 C2107 C2108 C2109 C2110 C2111
Generals
 X[0,0] X[1,0] X[2,0] X[3,0] X[4,0] X[5,0] X[6,0] X[7,0] X[8,0] X[9,0]
General Constraints
 embedding_magnitude_constraint: embedding_magnitude = NORM ( 2 ) (
   Aggregation[0] , Aggregation[1] , Aggregation[2] , Aggregation[3] ,
   Aggregation[4] , Aggregation[5] , Aggregation[6] , Aggregation[7] )
End
