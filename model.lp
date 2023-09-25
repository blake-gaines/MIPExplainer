\ Model GNN Inverse
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  Output_t[0] + 10 embedding_similarity
Subject To
 node_0_connected: A[0,1] + A[0,2] + A[0,3] + A[0,4] + A[1,0] + A[2,0]
   + A[3,0] + A[4,0] >= 1
 node_1_connected: A[1,2] + A[1,3] + A[1,4] + A[2,1] + A[3,1] + A[4,1]
   >= 1
 node_2_connected: A[2,3] + A[2,4] + A[3,2] + A[4,2] >= 1
 node_3_connected: A[3,4] + A[4,3] >= 1
 undirected_0_0: = 0
 undirected_0_1: A[0,1] - A[1,0] = 0
 undirected_0_2: A[0,2] - A[2,0] = 0
 undirected_0_3: A[0,3] - A[3,0] = 0
 undirected_0_4: A[0,4] - A[4,0] = 0
 undirected_1_1: = 0
 undirected_1_2: A[1,2] - A[2,1] = 0
 undirected_1_3: A[1,3] - A[3,1] = 0
 undirected_1_4: A[1,4] - A[4,1] = 0
 undirected_2_2: = 0
 undirected_2_3: A[2,3] - A[3,2] = 0
 undirected_2_4: A[2,4] - A[4,2] = 0
 undirected_3_3: = 0
 undirected_3_4: A[3,4] - A[4,3] = 0
 undirected_4_4: = 0
 non_isolatied[0]: A[0,0] + A[1,0] + A[2,0] + A[3,0] + A[4,0] >= 1
 non_isolatied[1]: A[0,1] + A[1,1] + A[2,1] + A[3,1] + A[4,1] >= 1
 non_isolatied[2]: A[0,2] + A[1,2] + A[2,2] + A[3,2] + A[4,2] >= 1
 non_isolatied[3]: A[0,3] + A[1,3] + A[2,3] + A[3,3] + A[4,3] >= 1
 non_isolatied[4]: A[0,4] + A[1,4] + A[2,4] + A[3,4] + A[4,4] >= 1
 features_are_node_degrees[0,0]: - A[0,0] - A[1,0] - A[2,0] - A[3,0]
   - A[4,0] + X[0,0] = 0
 features_are_node_degrees[1,0]: - A[0,1] - A[1,1] - A[2,1] - A[3,1]
   - A[4,1] + X[1,0] = 0
 features_are_node_degrees[2,0]: - A[0,2] - A[1,2] - A[2,2] - A[3,2]
   - A[4,2] + X[2,0] = 0
 features_are_node_degrees[3,0]: - A[0,3] - A[1,3] - A[2,3] - A[3,3]
   - A[4,3] + X[3,0] = 0
 features_are_node_degrees[4,0]: - A[0,4] - A[1,4] - A[2,4] - A[3,4]
   - A[4,4] + X[4,0] = 0
 Conv_0_output_constraint[0,0]: 1.0302494764328 X[0,0]
   - 1.753483533859253 Conv_0_aggregated_features[0,0] + Conv_0_t[0,0]
   = -0.4682744145393372
 Conv_0_output_constraint[0,1]: 0.1200120449066162 X[0,0]
   + 0.4937304258346558 Conv_0_aggregated_features[0,0] + Conv_0_t[0,1]
   = 0.4476426839828491
 Conv_0_output_constraint[0,2]: 0.8109501600265503 X[0,0]
   + 0.5733267068862915 Conv_0_aggregated_features[0,0] + Conv_0_t[0,2]
   = -0.372223973274231
 Conv_0_output_constraint[0,3]: - 1.394621968269348 X[0,0]
   + 0.2061941474676132 Conv_0_aggregated_features[0,0] + Conv_0_t[0,3]
   = -1.02570378780365
 Conv_0_output_constraint[1,0]: 1.0302494764328 X[1,0]
   - 1.753483533859253 Conv_0_aggregated_features[1,0] + Conv_0_t[1,0]
   = -0.4682744145393372
 Conv_0_output_constraint[1,1]: 0.1200120449066162 X[1,0]
   + 0.4937304258346558 Conv_0_aggregated_features[1,0] + Conv_0_t[1,1]
   = 0.4476426839828491
 Conv_0_output_constraint[1,2]: 0.8109501600265503 X[1,0]
   + 0.5733267068862915 Conv_0_aggregated_features[1,0] + Conv_0_t[1,2]
   = -0.372223973274231
 Conv_0_output_constraint[1,3]: - 1.394621968269348 X[1,0]
   + 0.2061941474676132 Conv_0_aggregated_features[1,0] + Conv_0_t[1,3]
   = -1.02570378780365
 Conv_0_output_constraint[2,0]: 1.0302494764328 X[2,0]
   - 1.753483533859253 Conv_0_aggregated_features[2,0] + Conv_0_t[2,0]
   = -0.4682744145393372
 Conv_0_output_constraint[2,1]: 0.1200120449066162 X[2,0]
   + 0.4937304258346558 Conv_0_aggregated_features[2,0] + Conv_0_t[2,1]
   = 0.4476426839828491
 Conv_0_output_constraint[2,2]: 0.8109501600265503 X[2,0]
   + 0.5733267068862915 Conv_0_aggregated_features[2,0] + Conv_0_t[2,2]
   = -0.372223973274231
 Conv_0_output_constraint[2,3]: - 1.394621968269348 X[2,0]
   + 0.2061941474676132 Conv_0_aggregated_features[2,0] + Conv_0_t[2,3]
   = -1.02570378780365
 Conv_0_output_constraint[3,0]: 1.0302494764328 X[3,0]
   - 1.753483533859253 Conv_0_aggregated_features[3,0] + Conv_0_t[3,0]
   = -0.4682744145393372
 Conv_0_output_constraint[3,1]: 0.1200120449066162 X[3,0]
   + 0.4937304258346558 Conv_0_aggregated_features[3,0] + Conv_0_t[3,1]
   = 0.4476426839828491
 Conv_0_output_constraint[3,2]: 0.8109501600265503 X[3,0]
   + 0.5733267068862915 Conv_0_aggregated_features[3,0] + Conv_0_t[3,2]
   = -0.372223973274231
 Conv_0_output_constraint[3,3]: - 1.394621968269348 X[3,0]
   + 0.2061941474676132 Conv_0_aggregated_features[3,0] + Conv_0_t[3,3]
   = -1.02570378780365
 Conv_0_output_constraint[4,0]: 1.0302494764328 X[4,0]
   - 1.753483533859253 Conv_0_aggregated_features[4,0] + Conv_0_t[4,0]
   = -0.4682744145393372
 Conv_0_output_constraint[4,1]: 0.1200120449066162 X[4,0]
   + 0.4937304258346558 Conv_0_aggregated_features[4,0] + Conv_0_t[4,1]
   = 0.4476426839828491
 Conv_0_output_constraint[4,2]: 0.8109501600265503 X[4,0]
   + 0.5733267068862915 Conv_0_aggregated_features[4,0] + Conv_0_t[4,2]
   = -0.372223973274231
 Conv_0_output_constraint[4,3]: - 1.394621968269348 X[4,0]
   + 0.2061941474676132 Conv_0_aggregated_features[4,0] + Conv_0_t[4,3]
   = -1.02570378780365
 Conv_0_Relu_constraint_1[0,0]: - Conv_0_t[0,0] + Conv_0_Relu_ts[0,0]
   - Conv_0_Relu_ss[0,0] = 0
 Conv_0_Relu_constraint_1[0,1]: - Conv_0_t[0,1] + Conv_0_Relu_ts[0,1]
   - Conv_0_Relu_ss[0,1] = 0
 Conv_0_Relu_constraint_1[0,2]: - Conv_0_t[0,2] + Conv_0_Relu_ts[0,2]
   - Conv_0_Relu_ss[0,2] = 0
 Conv_0_Relu_constraint_1[0,3]: - Conv_0_t[0,3] + Conv_0_Relu_ts[0,3]
   - Conv_0_Relu_ss[0,3] = 0
 Conv_0_Relu_constraint_1[1,0]: - Conv_0_t[1,0] + Conv_0_Relu_ts[1,0]
   - Conv_0_Relu_ss[1,0] = 0
 Conv_0_Relu_constraint_1[1,1]: - Conv_0_t[1,1] + Conv_0_Relu_ts[1,1]
   - Conv_0_Relu_ss[1,1] = 0
 Conv_0_Relu_constraint_1[1,2]: - Conv_0_t[1,2] + Conv_0_Relu_ts[1,2]
   - Conv_0_Relu_ss[1,2] = 0
 Conv_0_Relu_constraint_1[1,3]: - Conv_0_t[1,3] + Conv_0_Relu_ts[1,3]
   - Conv_0_Relu_ss[1,3] = 0
 Conv_0_Relu_constraint_1[2,0]: - Conv_0_t[2,0] + Conv_0_Relu_ts[2,0]
   - Conv_0_Relu_ss[2,0] = 0
 Conv_0_Relu_constraint_1[2,1]: - Conv_0_t[2,1] + Conv_0_Relu_ts[2,1]
   - Conv_0_Relu_ss[2,1] = 0
 Conv_0_Relu_constraint_1[2,2]: - Conv_0_t[2,2] + Conv_0_Relu_ts[2,2]
   - Conv_0_Relu_ss[2,2] = 0
 Conv_0_Relu_constraint_1[2,3]: - Conv_0_t[2,3] + Conv_0_Relu_ts[2,3]
   - Conv_0_Relu_ss[2,3] = 0
 Conv_0_Relu_constraint_1[3,0]: - Conv_0_t[3,0] + Conv_0_Relu_ts[3,0]
   - Conv_0_Relu_ss[3,0] = 0
 Conv_0_Relu_constraint_1[3,1]: - Conv_0_t[3,1] + Conv_0_Relu_ts[3,1]
   - Conv_0_Relu_ss[3,1] = 0
 Conv_0_Relu_constraint_1[3,2]: - Conv_0_t[3,2] + Conv_0_Relu_ts[3,2]
   - Conv_0_Relu_ss[3,2] = 0
 Conv_0_Relu_constraint_1[3,3]: - Conv_0_t[3,3] + Conv_0_Relu_ts[3,3]
   - Conv_0_Relu_ss[3,3] = 0
 Conv_0_Relu_constraint_1[4,0]: - Conv_0_t[4,0] + Conv_0_Relu_ts[4,0]
   - Conv_0_Relu_ss[4,0] = 0
 Conv_0_Relu_constraint_1[4,1]: - Conv_0_t[4,1] + Conv_0_Relu_ts[4,1]
   - Conv_0_Relu_ss[4,1] = 0
 Conv_0_Relu_constraint_1[4,2]: - Conv_0_t[4,2] + Conv_0_Relu_ts[4,2]
   - Conv_0_Relu_ss[4,2] = 0
 Conv_0_Relu_constraint_1[4,3]: - Conv_0_t[4,3] + Conv_0_Relu_ts[4,3]
   - Conv_0_Relu_ss[4,3] = 0
 Conv_0_Relu_constraint_2[0,0]: Conv_0_Relu_ts[0,0] - 8.299143254756927 C95
   <= 0
 Conv_0_Relu_constraint_2[0,1]: Conv_0_Relu_ts[0,1]
   - 0.4476426839828491 C96 <= 0
 Conv_0_Relu_constraint_2[0,2]: Conv_0_Relu_ts[0,2] <= 0
 Conv_0_Relu_constraint_2[0,3]: Conv_0_Relu_ts[0,3] - 5.947406053543091 C98
   <= 0
 Conv_0_Relu_constraint_2[1,0]: Conv_0_Relu_ts[1,0] - 8.299143254756927 C99
   <= 0
 Conv_0_Relu_constraint_2[1,1]: Conv_0_Relu_ts[1,1]
   - 0.4476426839828491 C100 <= 0
 Conv_0_Relu_constraint_2[1,2]: Conv_0_Relu_ts[1,2] <= 0
 Conv_0_Relu_constraint_2[1,3]: Conv_0_Relu_ts[1,3]
   - 5.947406053543091 C102 <= 0
 Conv_0_Relu_constraint_2[2,0]: Conv_0_Relu_ts[2,0]
   - 8.299143254756927 C103 <= 0
 Conv_0_Relu_constraint_2[2,1]: Conv_0_Relu_ts[2,1]
   - 0.4476426839828491 C104 <= 0
 Conv_0_Relu_constraint_2[2,2]: Conv_0_Relu_ts[2,2] <= 0
 Conv_0_Relu_constraint_2[2,3]: Conv_0_Relu_ts[2,3]
   - 5.947406053543091 C106 <= 0
 Conv_0_Relu_constraint_2[3,0]: Conv_0_Relu_ts[3,0]
   - 8.299143254756927 C107 <= 0
 Conv_0_Relu_constraint_2[3,1]: Conv_0_Relu_ts[3,1]
   - 0.4476426839828491 C108 <= 0
 Conv_0_Relu_constraint_2[3,2]: Conv_0_Relu_ts[3,2] <= 0
 Conv_0_Relu_constraint_2[3,3]: Conv_0_Relu_ts[3,3]
   - 5.947406053543091 C110 <= 0
 Conv_0_Relu_constraint_2[4,0]: Conv_0_Relu_ts[4,0]
   - 8.299143254756927 C111 <= 0
 Conv_0_Relu_constraint_2[4,1]: Conv_0_Relu_ts[4,1]
   - 0.4476426839828491 C112 <= 0
 Conv_0_Relu_constraint_2[4,2]: Conv_0_Relu_ts[4,2] <= 0
 Conv_0_Relu_constraint_2[4,3]: Conv_0_Relu_ts[4,3]
   - 5.947406053543091 C114 <= 0
 Conv_0_Relu_constraint_3[0,0]: Conv_0_Relu_ss[0,0] + 5.619521796703339 C95
   <= 5.619521796703339
 Conv_0_Relu_constraint_3[0,1]: Conv_0_Relu_ss[0,1] + 2.621069669723511 C96
   <= 2.621069669723511
 Conv_0_Relu_constraint_3[0,2]: Conv_0_Relu_ss[0,2] + 7.29360830783844 C97
   <= 7.29360830783844
 Conv_0_Relu_constraint_3[0,3]: Conv_0_Relu_ss[0,3] + 2.056674525141716 C98
   <= 2.056674525141716
 Conv_0_Relu_constraint_3[1,0]: Conv_0_Relu_ss[1,0] + 5.619521796703339 C99
   <= 5.619521796703339
 Conv_0_Relu_constraint_3[1,1]: Conv_0_Relu_ss[1,1]
   + 2.621069669723511 C100 <= 2.621069669723511
 Conv_0_Relu_constraint_3[1,2]: Conv_0_Relu_ss[1,2] + 7.29360830783844 C101
   <= 7.29360830783844
 Conv_0_Relu_constraint_3[1,3]: Conv_0_Relu_ss[1,3]
   + 2.056674525141716 C102 <= 2.056674525141716
 Conv_0_Relu_constraint_3[2,0]: Conv_0_Relu_ss[2,0]
   + 5.619521796703339 C103 <= 5.619521796703339
 Conv_0_Relu_constraint_3[2,1]: Conv_0_Relu_ss[2,1]
   + 2.621069669723511 C104 <= 2.621069669723511
 Conv_0_Relu_constraint_3[2,2]: Conv_0_Relu_ss[2,2] + 7.29360830783844 C105
   <= 7.29360830783844
 Conv_0_Relu_constraint_3[2,3]: Conv_0_Relu_ss[2,3]
   + 2.056674525141716 C106 <= 2.056674525141716
 Conv_0_Relu_constraint_3[3,0]: Conv_0_Relu_ss[3,0]
   + 5.619521796703339 C107 <= 5.619521796703339
 Conv_0_Relu_constraint_3[3,1]: Conv_0_Relu_ss[3,1]
   + 2.621069669723511 C108 <= 2.621069669723511
 Conv_0_Relu_constraint_3[3,2]: Conv_0_Relu_ss[3,2] + 7.29360830783844 C109
   <= 7.29360830783844
 Conv_0_Relu_constraint_3[3,3]: Conv_0_Relu_ss[3,3]
   + 2.056674525141716 C110 <= 2.056674525141716
 Conv_0_Relu_constraint_3[4,0]: Conv_0_Relu_ss[4,0]
   + 5.619521796703339 C111 <= 5.619521796703339
 Conv_0_Relu_constraint_3[4,1]: Conv_0_Relu_ss[4,1]
   + 2.621069669723511 C112 <= 2.621069669723511
 Conv_0_Relu_constraint_3[4,2]: Conv_0_Relu_ss[4,2] + 7.29360830783844 C113
   <= 7.29360830783844
 Conv_0_Relu_constraint_3[4,3]: Conv_0_Relu_ss[4,3]
   + 2.056674525141716 C114 <= 2.056674525141716
 Conv_1_output_constraint[0,0]: - 0.5920342803001404 Conv_0_Relu_ts[0,0]
   - 0.3072851896286011 Conv_0_Relu_ts[0,1]
   + 0.4429754614830017 Conv_0_Relu_ts[0,2]
   + 1.245180368423462 Conv_0_Relu_ts[0,3]
   + 1.295092701911926 Conv_1_aggregated_features[0,0]
   - 0.0124646425247192 Conv_1_aggregated_features[0,1]
   - 0.3947818279266357 Conv_1_aggregated_features[0,2]
   - 0.6485974788665771 Conv_1_aggregated_features[0,3] + Conv_1_t[0,0]
   = 0.7127057313919067
 Conv_1_output_constraint[0,1]: - 0.2783041298389435 Conv_0_Relu_ts[0,0]
   + 0.4955700635910034 Conv_0_Relu_ts[0,1]
   - 0.0679431557655334 Conv_0_Relu_ts[0,2]
   + 0.7996922731399536 Conv_0_Relu_ts[0,3]
   + 0.0373080149292946 Conv_1_aggregated_features[0,0]
   + 0.0989940762519836 Conv_1_aggregated_features[0,1]
   + 0.1828911304473877 Conv_1_aggregated_features[0,2]
   + 0.6289404034614563 Conv_1_aggregated_features[0,3] + Conv_1_t[0,1]
   = 1.530361175537109
 Conv_1_output_constraint[0,2]: 1.680143475532532 Conv_0_Relu_ts[0,0]
   + 0.2031083703041077 Conv_0_Relu_ts[0,1]
   + 0.267920196056366 Conv_0_Relu_ts[0,2]
   + 0.0116705177351832 Conv_0_Relu_ts[0,3]
   - 1.532716155052185 Conv_1_aggregated_features[0,0]
   - 0.466221809387207 Conv_1_aggregated_features[0,1]
   - 0.2213227152824402 Conv_1_aggregated_features[0,2]
   + 0.7102178931236267 Conv_1_aggregated_features[0,3] + Conv_1_t[0,2]
   = 0.4375678300857544
 Conv_1_output_constraint[0,3]: - 0.0998657420277596 Conv_0_Relu_ts[0,0]
   + 0.1217613816261292 Conv_0_Relu_ts[0,1]
   - 0.0995404124259949 Conv_0_Relu_ts[0,2]
   - 1.026525974273682 Conv_0_Relu_ts[0,3]
   - 1.344330072402954 Conv_1_aggregated_features[0,0]
   - 0.0778789520263672 Conv_1_aggregated_features[0,1]
   + 0.1988690495491028 Conv_1_aggregated_features[0,2]
   - 0.4722947776317596 Conv_1_aggregated_features[0,3] + Conv_1_t[0,3]
   = -1.126737356185913
 Conv_1_output_constraint[1,0]: - 0.5920342803001404 Conv_0_Relu_ts[1,0]
   - 0.3072851896286011 Conv_0_Relu_ts[1,1]
   + 0.4429754614830017 Conv_0_Relu_ts[1,2]
   + 1.245180368423462 Conv_0_Relu_ts[1,3]
   + 1.295092701911926 Conv_1_aggregated_features[1,0]
   - 0.0124646425247192 Conv_1_aggregated_features[1,1]
   - 0.3947818279266357 Conv_1_aggregated_features[1,2]
   - 0.6485974788665771 Conv_1_aggregated_features[1,3] + Conv_1_t[1,0]
   = 0.7127057313919067
 Conv_1_output_constraint[1,1]: - 0.2783041298389435 Conv_0_Relu_ts[1,0]
   + 0.4955700635910034 Conv_0_Relu_ts[1,1]
   - 0.0679431557655334 Conv_0_Relu_ts[1,2]
   + 0.7996922731399536 Conv_0_Relu_ts[1,3]
   + 0.0373080149292946 Conv_1_aggregated_features[1,0]
   + 0.0989940762519836 Conv_1_aggregated_features[1,1]
   + 0.1828911304473877 Conv_1_aggregated_features[1,2]
   + 0.6289404034614563 Conv_1_aggregated_features[1,3] + Conv_1_t[1,1]
   = 1.530361175537109
 Conv_1_output_constraint[1,2]: 1.680143475532532 Conv_0_Relu_ts[1,0]
   + 0.2031083703041077 Conv_0_Relu_ts[1,1]
   + 0.267920196056366 Conv_0_Relu_ts[1,2]
   + 0.0116705177351832 Conv_0_Relu_ts[1,3]
   - 1.532716155052185 Conv_1_aggregated_features[1,0]
   - 0.466221809387207 Conv_1_aggregated_features[1,1]
   - 0.2213227152824402 Conv_1_aggregated_features[1,2]
   + 0.7102178931236267 Conv_1_aggregated_features[1,3] + Conv_1_t[1,2]
   = 0.4375678300857544
 Conv_1_output_constraint[1,3]: - 0.0998657420277596 Conv_0_Relu_ts[1,0]
   + 0.1217613816261292 Conv_0_Relu_ts[1,1]
   - 0.0995404124259949 Conv_0_Relu_ts[1,2]
   - 1.026525974273682 Conv_0_Relu_ts[1,3]
   - 1.344330072402954 Conv_1_aggregated_features[1,0]
   - 0.0778789520263672 Conv_1_aggregated_features[1,1]
   + 0.1988690495491028 Conv_1_aggregated_features[1,2]
   - 0.4722947776317596 Conv_1_aggregated_features[1,3] + Conv_1_t[1,3]
   = -1.126737356185913
 Conv_1_output_constraint[2,0]: - 0.5920342803001404 Conv_0_Relu_ts[2,0]
   - 0.3072851896286011 Conv_0_Relu_ts[2,1]
   + 0.4429754614830017 Conv_0_Relu_ts[2,2]
   + 1.245180368423462 Conv_0_Relu_ts[2,3]
   + 1.295092701911926 Conv_1_aggregated_features[2,0]
   - 0.0124646425247192 Conv_1_aggregated_features[2,1]
   - 0.3947818279266357 Conv_1_aggregated_features[2,2]
   - 0.6485974788665771 Conv_1_aggregated_features[2,3] + Conv_1_t[2,0]
   = 0.7127057313919067
 Conv_1_output_constraint[2,1]: - 0.2783041298389435 Conv_0_Relu_ts[2,0]
   + 0.4955700635910034 Conv_0_Relu_ts[2,1]
   - 0.0679431557655334 Conv_0_Relu_ts[2,2]
   + 0.7996922731399536 Conv_0_Relu_ts[2,3]
   + 0.0373080149292946 Conv_1_aggregated_features[2,0]
   + 0.0989940762519836 Conv_1_aggregated_features[2,1]
   + 0.1828911304473877 Conv_1_aggregated_features[2,2]
   + 0.6289404034614563 Conv_1_aggregated_features[2,3] + Conv_1_t[2,1]
   = 1.530361175537109
 Conv_1_output_constraint[2,2]: 1.680143475532532 Conv_0_Relu_ts[2,0]
   + 0.2031083703041077 Conv_0_Relu_ts[2,1]
   + 0.267920196056366 Conv_0_Relu_ts[2,2]
   + 0.0116705177351832 Conv_0_Relu_ts[2,3]
   - 1.532716155052185 Conv_1_aggregated_features[2,0]
   - 0.466221809387207 Conv_1_aggregated_features[2,1]
   - 0.2213227152824402 Conv_1_aggregated_features[2,2]
   + 0.7102178931236267 Conv_1_aggregated_features[2,3] + Conv_1_t[2,2]
   = 0.4375678300857544
 Conv_1_output_constraint[2,3]: - 0.0998657420277596 Conv_0_Relu_ts[2,0]
   + 0.1217613816261292 Conv_0_Relu_ts[2,1]
   - 0.0995404124259949 Conv_0_Relu_ts[2,2]
   - 1.026525974273682 Conv_0_Relu_ts[2,3]
   - 1.344330072402954 Conv_1_aggregated_features[2,0]
   - 0.0778789520263672 Conv_1_aggregated_features[2,1]
   + 0.1988690495491028 Conv_1_aggregated_features[2,2]
   - 0.4722947776317596 Conv_1_aggregated_features[2,3] + Conv_1_t[2,3]
   = -1.126737356185913
 Conv_1_output_constraint[3,0]: - 0.5920342803001404 Conv_0_Relu_ts[3,0]
   - 0.3072851896286011 Conv_0_Relu_ts[3,1]
   + 0.4429754614830017 Conv_0_Relu_ts[3,2]
   + 1.245180368423462 Conv_0_Relu_ts[3,3]
   + 1.295092701911926 Conv_1_aggregated_features[3,0]
   - 0.0124646425247192 Conv_1_aggregated_features[3,1]
   - 0.3947818279266357 Conv_1_aggregated_features[3,2]
   - 0.6485974788665771 Conv_1_aggregated_features[3,3] + Conv_1_t[3,0]
   = 0.7127057313919067
 Conv_1_output_constraint[3,1]: - 0.2783041298389435 Conv_0_Relu_ts[3,0]
   + 0.4955700635910034 Conv_0_Relu_ts[3,1]
   - 0.0679431557655334 Conv_0_Relu_ts[3,2]
   + 0.7996922731399536 Conv_0_Relu_ts[3,3]
   + 0.0373080149292946 Conv_1_aggregated_features[3,0]
   + 0.0989940762519836 Conv_1_aggregated_features[3,1]
   + 0.1828911304473877 Conv_1_aggregated_features[3,2]
   + 0.6289404034614563 Conv_1_aggregated_features[3,3] + Conv_1_t[3,1]
   = 1.530361175537109
 Conv_1_output_constraint[3,2]: 1.680143475532532 Conv_0_Relu_ts[3,0]
   + 0.2031083703041077 Conv_0_Relu_ts[3,1]
   + 0.267920196056366 Conv_0_Relu_ts[3,2]
   + 0.0116705177351832 Conv_0_Relu_ts[3,3]
   - 1.532716155052185 Conv_1_aggregated_features[3,0]
   - 0.466221809387207 Conv_1_aggregated_features[3,1]
   - 0.2213227152824402 Conv_1_aggregated_features[3,2]
   + 0.7102178931236267 Conv_1_aggregated_features[3,3] + Conv_1_t[3,2]
   = 0.4375678300857544
 Conv_1_output_constraint[3,3]: - 0.0998657420277596 Conv_0_Relu_ts[3,0]
   + 0.1217613816261292 Conv_0_Relu_ts[3,1]
   - 0.0995404124259949 Conv_0_Relu_ts[3,2]
   - 1.026525974273682 Conv_0_Relu_ts[3,3]
   - 1.344330072402954 Conv_1_aggregated_features[3,0]
   - 0.0778789520263672 Conv_1_aggregated_features[3,1]
   + 0.1988690495491028 Conv_1_aggregated_features[3,2]
   - 0.4722947776317596 Conv_1_aggregated_features[3,3] + Conv_1_t[3,3]
   = -1.126737356185913
 Conv_1_output_constraint[4,0]: - 0.5920342803001404 Conv_0_Relu_ts[4,0]
   - 0.3072851896286011 Conv_0_Relu_ts[4,1]
   + 0.4429754614830017 Conv_0_Relu_ts[4,2]
   + 1.245180368423462 Conv_0_Relu_ts[4,3]
   + 1.295092701911926 Conv_1_aggregated_features[4,0]
   - 0.0124646425247192 Conv_1_aggregated_features[4,1]
   - 0.3947818279266357 Conv_1_aggregated_features[4,2]
   - 0.6485974788665771 Conv_1_aggregated_features[4,3] + Conv_1_t[4,0]
   = 0.7127057313919067
 Conv_1_output_constraint[4,1]: - 0.2783041298389435 Conv_0_Relu_ts[4,0]
   + 0.4955700635910034 Conv_0_Relu_ts[4,1]
   - 0.0679431557655334 Conv_0_Relu_ts[4,2]
   + 0.7996922731399536 Conv_0_Relu_ts[4,3]
   + 0.0373080149292946 Conv_1_aggregated_features[4,0]
   + 0.0989940762519836 Conv_1_aggregated_features[4,1]
   + 0.1828911304473877 Conv_1_aggregated_features[4,2]
   + 0.6289404034614563 Conv_1_aggregated_features[4,3] + Conv_1_t[4,1]
   = 1.530361175537109
 Conv_1_output_constraint[4,2]: 1.680143475532532 Conv_0_Relu_ts[4,0]
   + 0.2031083703041077 Conv_0_Relu_ts[4,1]
   + 0.267920196056366 Conv_0_Relu_ts[4,2]
   + 0.0116705177351832 Conv_0_Relu_ts[4,3]
   - 1.532716155052185 Conv_1_aggregated_features[4,0]
   - 0.466221809387207 Conv_1_aggregated_features[4,1]
   - 0.2213227152824402 Conv_1_aggregated_features[4,2]
   + 0.7102178931236267 Conv_1_aggregated_features[4,3] + Conv_1_t[4,2]
   = 0.4375678300857544
 Conv_1_output_constraint[4,3]: - 0.0998657420277596 Conv_0_Relu_ts[4,0]
   + 0.1217613816261292 Conv_0_Relu_ts[4,1]
   - 0.0995404124259949 Conv_0_Relu_ts[4,2]
   - 1.026525974273682 Conv_0_Relu_ts[4,3]
   - 1.344330072402954 Conv_1_aggregated_features[4,0]
   - 0.0778789520263672 Conv_1_aggregated_features[4,1]
   + 0.1988690495491028 Conv_1_aggregated_features[4,2]
   - 0.4722947776317596 Conv_1_aggregated_features[4,3] + Conv_1_t[4,3]
   = -1.126737356185913
 Conv_1_Relu_constraint_1[0,0]: - Conv_1_t[0,0] + Conv_1_Relu_ts[0,0]
   - Conv_1_Relu_ss[0,0] = 0
 Conv_1_Relu_constraint_1[0,1]: - Conv_1_t[0,1] + Conv_1_Relu_ts[0,1]
   - Conv_1_Relu_ss[0,1] = 0
 Conv_1_Relu_constraint_1[0,2]: - Conv_1_t[0,2] + Conv_1_Relu_ts[0,2]
   - Conv_1_Relu_ss[0,2] = 0
 Conv_1_Relu_constraint_1[0,3]: - Conv_1_t[0,3] + Conv_1_Relu_ts[0,3]
   - Conv_1_Relu_ss[0,3] = 0
 Conv_1_Relu_constraint_1[1,0]: - Conv_1_t[1,0] + Conv_1_Relu_ts[1,0]
   - Conv_1_Relu_ss[1,0] = 0
 Conv_1_Relu_constraint_1[1,1]: - Conv_1_t[1,1] + Conv_1_Relu_ts[1,1]
   - Conv_1_Relu_ss[1,1] = 0
 Conv_1_Relu_constraint_1[1,2]: - Conv_1_t[1,2] + Conv_1_Relu_ts[1,2]
   - Conv_1_Relu_ss[1,2] = 0
 Conv_1_Relu_constraint_1[1,3]: - Conv_1_t[1,3] + Conv_1_Relu_ts[1,3]
   - Conv_1_Relu_ss[1,3] = 0
 Conv_1_Relu_constraint_1[2,0]: - Conv_1_t[2,0] + Conv_1_Relu_ts[2,0]
   - Conv_1_Relu_ss[2,0] = 0
 Conv_1_Relu_constraint_1[2,1]: - Conv_1_t[2,1] + Conv_1_Relu_ts[2,1]
   - Conv_1_Relu_ss[2,1] = 0
 Conv_1_Relu_constraint_1[2,2]: - Conv_1_t[2,2] + Conv_1_Relu_ts[2,2]
   - Conv_1_Relu_ss[2,2] = 0
 Conv_1_Relu_constraint_1[2,3]: - Conv_1_t[2,3] + Conv_1_Relu_ts[2,3]
   - Conv_1_Relu_ss[2,3] = 0
 Conv_1_Relu_constraint_1[3,0]: - Conv_1_t[3,0] + Conv_1_Relu_ts[3,0]
   - Conv_1_Relu_ss[3,0] = 0
 Conv_1_Relu_constraint_1[3,1]: - Conv_1_t[3,1] + Conv_1_Relu_ts[3,1]
   - Conv_1_Relu_ss[3,1] = 0
 Conv_1_Relu_constraint_1[3,2]: - Conv_1_t[3,2] + Conv_1_Relu_ts[3,2]
   - Conv_1_Relu_ss[3,2] = 0
 Conv_1_Relu_constraint_1[3,3]: - Conv_1_t[3,3] + Conv_1_Relu_ts[3,3]
   - Conv_1_Relu_ss[3,3] = 0
 Conv_1_Relu_constraint_1[4,0]: - Conv_1_t[4,0] + Conv_1_Relu_ts[4,0]
   - Conv_1_Relu_ss[4,0] = 0
 Conv_1_Relu_constraint_1[4,1]: - Conv_1_t[4,1] + Conv_1_Relu_ts[4,1]
   - Conv_1_Relu_ss[4,1] = 0
 Conv_1_Relu_constraint_1[4,2]: - Conv_1_t[4,2] + Conv_1_Relu_ts[4,2]
   - Conv_1_Relu_ss[4,2] = 0
 Conv_1_Relu_constraint_1[4,3]: - Conv_1_t[4,3] + Conv_1_Relu_ts[4,3]
   - Conv_1_Relu_ss[4,3] = 0
 Conv_1_Relu_constraint_2[0,0]: Conv_1_Relu_ts[0,0]
   - 9.626689280521735 C195 <= 0
 Conv_1_Relu_constraint_2[0,1]: Conv_1_Relu_ts[0,1]
   - 3.840047017460973 C196 <= 0
 Conv_1_Relu_constraint_2[0,2]: Conv_1_Relu_ts[0,2] - 13.3664995518295 C197
   <= 0
 Conv_1_Relu_constraint_2[0,3]: Conv_1_Relu_ts[0,3]
   - 19.80780815186422 C198 <= 0
 Conv_1_Relu_constraint_2[1,0]: Conv_1_Relu_ts[1,0]
   - 9.626689280521735 C199 <= 0
 Conv_1_Relu_constraint_2[1,1]: Conv_1_Relu_ts[1,1]
   - 3.840047017460973 C200 <= 0
 Conv_1_Relu_constraint_2[1,2]: Conv_1_Relu_ts[1,2] - 13.3664995518295 C201
   <= 0
 Conv_1_Relu_constraint_2[1,3]: Conv_1_Relu_ts[1,3]
   - 19.80780815186422 C202 <= 0
 Conv_1_Relu_constraint_2[2,0]: Conv_1_Relu_ts[2,0]
   - 9.626689280521735 C203 <= 0
 Conv_1_Relu_constraint_2[2,1]: Conv_1_Relu_ts[2,1]
   - 3.840047017460973 C204 <= 0
 Conv_1_Relu_constraint_2[2,2]: Conv_1_Relu_ts[2,2] - 13.3664995518295 C205
   <= 0
 Conv_1_Relu_constraint_2[2,3]: Conv_1_Relu_ts[2,3]
   - 19.80780815186422 C206 <= 0
 Conv_1_Relu_constraint_2[3,0]: Conv_1_Relu_ts[3,0]
   - 9.626689280521735 C207 <= 0
 Conv_1_Relu_constraint_2[3,1]: Conv_1_Relu_ts[3,1]
   - 3.840047017460973 C208 <= 0
 Conv_1_Relu_constraint_2[3,2]: Conv_1_Relu_ts[3,2] - 13.3664995518295 C209
   <= 0
 Conv_1_Relu_constraint_2[3,3]: Conv_1_Relu_ts[3,3]
   - 19.80780815186422 C210 <= 0
 Conv_1_Relu_constraint_2[4,0]: Conv_1_Relu_ts[4,0]
   - 9.626689280521735 C211 <= 0
 Conv_1_Relu_constraint_2[4,1]: Conv_1_Relu_ts[4,1]
   - 3.840047017460973 C212 <= 0
 Conv_1_Relu_constraint_2[4,2]: Conv_1_Relu_ts[4,2] - 13.3664995518295 C213
   <= 0
 Conv_1_Relu_constraint_2[4,3]: Conv_1_Relu_ts[4,3]
   - 19.80780815186422 C214 <= 0
 Conv_1_Relu_constraint_3[0,0]: Conv_1_Relu_ss[0,0]
   + 17.44104739088009 C195 <= 17.44104739088009
 Conv_1_Relu_constraint_3[0,1]: Conv_1_Relu_ss[0,1] + 7.54207430137968 C196
   <= 7.54207430137968
 Conv_1_Relu_constraint_3[0,2]: Conv_1_Relu_ss[0,2]
   + 17.89046704265052 C197 <= 17.89046704265052
 Conv_1_Relu_constraint_3[0,3]: Conv_1_Relu_ss[0,3]
   + 1.181242947862494 C198 <= 1.181242947862494
 Conv_1_Relu_constraint_3[1,0]: Conv_1_Relu_ss[1,0]
   + 17.44104739088009 C199 <= 17.44104739088009
 Conv_1_Relu_constraint_3[1,1]: Conv_1_Relu_ss[1,1] + 7.54207430137968 C200
   <= 7.54207430137968
 Conv_1_Relu_constraint_3[1,2]: Conv_1_Relu_ss[1,2]
   + 17.89046704265052 C201 <= 17.89046704265052
 Conv_1_Relu_constraint_3[1,3]: Conv_1_Relu_ss[1,3]
   + 1.181242947862494 C202 <= 1.181242947862494
 Conv_1_Relu_constraint_3[2,0]: Conv_1_Relu_ss[2,0]
   + 17.44104739088009 C203 <= 17.44104739088009
 Conv_1_Relu_constraint_3[2,1]: Conv_1_Relu_ss[2,1] + 7.54207430137968 C204
   <= 7.54207430137968
 Conv_1_Relu_constraint_3[2,2]: Conv_1_Relu_ss[2,2]
   + 17.89046704265052 C205 <= 17.89046704265052
 Conv_1_Relu_constraint_3[2,3]: Conv_1_Relu_ss[2,3]
   + 1.181242947862494 C206 <= 1.181242947862494
 Conv_1_Relu_constraint_3[3,0]: Conv_1_Relu_ss[3,0]
   + 17.44104739088009 C207 <= 17.44104739088009
 Conv_1_Relu_constraint_3[3,1]: Conv_1_Relu_ss[3,1] + 7.54207430137968 C208
   <= 7.54207430137968
 Conv_1_Relu_constraint_3[3,2]: Conv_1_Relu_ss[3,2]
   + 17.89046704265052 C209 <= 17.89046704265052
 Conv_1_Relu_constraint_3[3,3]: Conv_1_Relu_ss[3,3]
   + 1.181242947862494 C210 <= 1.181242947862494
 Conv_1_Relu_constraint_3[4,0]: Conv_1_Relu_ss[4,0]
   + 17.44104739088009 C211 <= 17.44104739088009
 Conv_1_Relu_constraint_3[4,1]: Conv_1_Relu_ss[4,1] + 7.54207430137968 C212
   <= 7.54207430137968
 Conv_1_Relu_constraint_3[4,2]: Conv_1_Relu_ss[4,2]
   + 17.89046704265052 C213 <= 17.89046704265052
 Conv_1_Relu_constraint_3[4,3]: Conv_1_Relu_ss[4,3]
   + 1.181242947862494 C214 <= 1.181242947862494
 Aggregation_constraint[0]: - 0.2 Conv_1_Relu_ts[0,0]
   - 0.2 Conv_1_Relu_ts[1,0] - 0.2 Conv_1_Relu_ts[2,0]
   - 0.2 Conv_1_Relu_ts[3,0] - 0.2 Conv_1_Relu_ts[4,0] + Aggregation[0]
   = 0
 Aggregation_constraint[1]: - 0.2 Conv_1_Relu_ts[0,1]
   - 0.2 Conv_1_Relu_ts[1,1] - 0.2 Conv_1_Relu_ts[2,1]
   - 0.2 Conv_1_Relu_ts[3,1] - 0.2 Conv_1_Relu_ts[4,1] + Aggregation[1]
   = 0
 Aggregation_constraint[2]: - 0.2 Conv_1_Relu_ts[0,2]
   - 0.2 Conv_1_Relu_ts[1,2] - 0.2 Conv_1_Relu_ts[2,2]
   - 0.2 Conv_1_Relu_ts[3,2] - 0.2 Conv_1_Relu_ts[4,2] + Aggregation[2]
   = 0
 Aggregation_constraint[3]: - 0.2 Conv_1_Relu_ts[0,3]
   - 0.2 Conv_1_Relu_ts[1,3] - 0.2 Conv_1_Relu_ts[2,3]
   - 0.2 Conv_1_Relu_ts[3,3] - 0.2 Conv_1_Relu_ts[4,3] + Aggregation[3]
   = 0
 Lin_0_output_constraint[0,0]: 1.732562780380249 Aggregation[0]
   + 3.886651754379272 Aggregation[1] + 5.742121696472168 Aggregation[2]
   - 0.9851827621459961 Aggregation[3] + Lin_0_t[0] = -0.8308897018432617
 Lin_0_output_constraint[0,1]: 0.6284189224243164 Aggregation[0]
   + 0.3881502449512482 Aggregation[1] - 0.3289519548416138 Aggregation[2]
   + 0.240267887711525 Aggregation[3] + Lin_0_t[1] = 0.2732022702693939
 Lin_0_output_constraint[0,2]: - 1.466388940811157 Aggregation[0]
   + 1.763412594795227 Aggregation[1] - 3.346069097518921 Aggregation[2]
   + 0.1937138587236404 Aggregation[3] + Lin_0_t[2] = 0.8427364230155945
 Lin_0_output_constraint[0,3]: 0.4696234464645386 Aggregation[0]
   - 0.1120558381080627 Aggregation[1] + 0.0530245900154114 Aggregation[2]
   + 0.4852983951568604 Aggregation[3] + Lin_0_t[3] = -0.3053258061408997
 Lin_0_Relu_constraint_1[0,0]: - Lin_0_t[0] + Lin_0_Relu_ts[0,0]
   - Lin_0_Relu_ss[0,0] = 0
 Lin_0_Relu_constraint_1[0,1]: - Lin_0_t[1] + Lin_0_Relu_ts[0,1]
   - Lin_0_Relu_ss[0,1] = 0
 Lin_0_Relu_constraint_1[0,2]: - Lin_0_t[2] + Lin_0_Relu_ts[0,2]
   - Lin_0_Relu_ss[0,2] = 0
 Lin_0_Relu_constraint_1[0,3]: - Lin_0_t[3] + Lin_0_Relu_ts[0,3]
   - Lin_0_Relu_ss[0,3] = 0
 Lin_0_Relu_constraint_2[0,0]: Lin_0_Relu_ts[0,0] - 19.51431114711157 C231
   <= 0
 Lin_0_Relu_constraint_2[0,1]: Lin_0_Relu_ts[0,1] - 4.396936156963867 C232
   <= 0
 Lin_0_Relu_constraint_2[0,2]: Lin_0_Relu_ts[0,2] - 58.84170178995957 C233
   <= 0
 Lin_0_Relu_constraint_2[0,3]: Lin_0_Relu_ts[0,3] - 0.430299686915956 C234
   <= 0
 Lin_0_Relu_constraint_3[0,0]: Lin_0_Relu_ss[0,0] + 108.3558361054767 C231
   <= 108.3558361054767
 Lin_0_Relu_constraint_3[0,1]: Lin_0_Relu_ss[0,1] + 12.29928911947451 C232
   <= 12.29928911947451
 Lin_0_Relu_constraint_3[0,2]: Lin_0_Relu_ss[0,2] + 10.60863422515173 C233
   <= 10.60863422515173
 Lin_0_Relu_constraint_3[0,3]: Lin_0_Relu_ss[0,3] + 14.84236966431347 C234
   <= 14.84236966431347
 Output_output_constraint[0,0]: 1.663121104240417 Lin_0_Relu_ts[0,0]
   - 0.268317848443985 Lin_0_Relu_ts[0,1]
   - 1.884008526802063 Lin_0_Relu_ts[0,2]
   - 0.3666214942932129 Lin_0_Relu_ts[0,3] + Output_t[0]
   = 0.4808872044086456
 Conv_0_averages_constraint[0,0]: [ - A[0,0] * X[0,0]
   + A[0,0] * Conv_0_aggregated_features[0,0] - A[0,1] * X[1,0]
   - A[0,2] * X[2,0] - A[0,3] * X[3,0] - A[0,4] * X[4,0]
   + A[1,0] * Conv_0_aggregated_features[0,0]
   + A[2,0] * Conv_0_aggregated_features[0,0]
   + A[3,0] * Conv_0_aggregated_features[0,0]
   + A[4,0] * Conv_0_aggregated_features[0,0] ] = 0
 Conv_0_averages_constraint[1,0]: [
   A[0,1] * Conv_0_aggregated_features[1,0] - A[1,0] * X[0,0]
   - A[1,1] * X[1,0] + A[1,1] * Conv_0_aggregated_features[1,0]
   - A[1,2] * X[2,0] - A[1,3] * X[3,0] - A[1,4] * X[4,0]
   + A[2,1] * Conv_0_aggregated_features[1,0]
   + A[3,1] * Conv_0_aggregated_features[1,0]
   + A[4,1] * Conv_0_aggregated_features[1,0] ] = 0
 Conv_0_averages_constraint[2,0]: [
   A[0,2] * Conv_0_aggregated_features[2,0]
   + A[1,2] * Conv_0_aggregated_features[2,0] - A[2,0] * X[0,0]
   - A[2,1] * X[1,0] - A[2,2] * X[2,0]
   + A[2,2] * Conv_0_aggregated_features[2,0] - A[2,3] * X[3,0]
   - A[2,4] * X[4,0] + A[3,2] * Conv_0_aggregated_features[2,0]
   + A[4,2] * Conv_0_aggregated_features[2,0] ] = 0
 Conv_0_averages_constraint[3,0]: [
   A[0,3] * Conv_0_aggregated_features[3,0]
   + A[1,3] * Conv_0_aggregated_features[3,0]
   + A[2,3] * Conv_0_aggregated_features[3,0] - A[3,0] * X[0,0]
   - A[3,1] * X[1,0] - A[3,2] * X[2,0] - A[3,3] * X[3,0]
   + A[3,3] * Conv_0_aggregated_features[3,0] - A[3,4] * X[4,0]
   + A[4,3] * Conv_0_aggregated_features[3,0] ] = 0
 Conv_0_averages_constraint[4,0]: [
   A[0,4] * Conv_0_aggregated_features[4,0]
   + A[1,4] * Conv_0_aggregated_features[4,0]
   + A[2,4] * Conv_0_aggregated_features[4,0]
   + A[3,4] * Conv_0_aggregated_features[4,0] - A[4,0] * X[0,0]
   - A[4,1] * X[1,0] - A[4,2] * X[2,0] - A[4,3] * X[3,0] - A[4,4] * X[4,0]
   + A[4,4] * Conv_0_aggregated_features[4,0] ] = 0
 Conv_1_averages_constraint[0,0]: [ - A[0,0] * Conv_0_Relu_ts[0,0]
   + A[0,0] * Conv_1_aggregated_features[0,0]
   - A[0,1] * Conv_0_Relu_ts[1,0] - A[0,2] * Conv_0_Relu_ts[2,0]
   - A[0,3] * Conv_0_Relu_ts[3,0] - A[0,4] * Conv_0_Relu_ts[4,0]
   + A[1,0] * Conv_1_aggregated_features[0,0]
   + A[2,0] * Conv_1_aggregated_features[0,0]
   + A[3,0] * Conv_1_aggregated_features[0,0]
   + A[4,0] * Conv_1_aggregated_features[0,0] ] = 0
 Conv_1_averages_constraint[0,1]: [ - A[0,0] * Conv_0_Relu_ts[0,1]
   + A[0,0] * Conv_1_aggregated_features[0,1]
   - A[0,1] * Conv_0_Relu_ts[1,1] - A[0,2] * Conv_0_Relu_ts[2,1]
   - A[0,3] * Conv_0_Relu_ts[3,1] - A[0,4] * Conv_0_Relu_ts[4,1]
   + A[1,0] * Conv_1_aggregated_features[0,1]
   + A[2,0] * Conv_1_aggregated_features[0,1]
   + A[3,0] * Conv_1_aggregated_features[0,1]
   + A[4,0] * Conv_1_aggregated_features[0,1] ] = 0
 Conv_1_averages_constraint[0,2]: [ - A[0,0] * Conv_0_Relu_ts[0,2]
   + A[0,0] * Conv_1_aggregated_features[0,2]
   - A[0,1] * Conv_0_Relu_ts[1,2] - A[0,2] * Conv_0_Relu_ts[2,2]
   - A[0,3] * Conv_0_Relu_ts[3,2] - A[0,4] * Conv_0_Relu_ts[4,2]
   + A[1,0] * Conv_1_aggregated_features[0,2]
   + A[2,0] * Conv_1_aggregated_features[0,2]
   + A[3,0] * Conv_1_aggregated_features[0,2]
   + A[4,0] * Conv_1_aggregated_features[0,2] ] = 0
 Conv_1_averages_constraint[0,3]: [ - A[0,0] * Conv_0_Relu_ts[0,3]
   + A[0,0] * Conv_1_aggregated_features[0,3]
   - A[0,1] * Conv_0_Relu_ts[1,3] - A[0,2] * Conv_0_Relu_ts[2,3]
   - A[0,3] * Conv_0_Relu_ts[3,3] - A[0,4] * Conv_0_Relu_ts[4,3]
   + A[1,0] * Conv_1_aggregated_features[0,3]
   + A[2,0] * Conv_1_aggregated_features[0,3]
   + A[3,0] * Conv_1_aggregated_features[0,3]
   + A[4,0] * Conv_1_aggregated_features[0,3] ] = 0
 Conv_1_averages_constraint[1,0]: [
   A[0,1] * Conv_1_aggregated_features[1,0] - A[1,0] * Conv_0_Relu_ts[0,0]
   - A[1,1] * Conv_0_Relu_ts[1,0]
   + A[1,1] * Conv_1_aggregated_features[1,0]
   - A[1,2] * Conv_0_Relu_ts[2,0] - A[1,3] * Conv_0_Relu_ts[3,0]
   - A[1,4] * Conv_0_Relu_ts[4,0]
   + A[2,1] * Conv_1_aggregated_features[1,0]
   + A[3,1] * Conv_1_aggregated_features[1,0]
   + A[4,1] * Conv_1_aggregated_features[1,0] ] = 0
 Conv_1_averages_constraint[1,1]: [
   A[0,1] * Conv_1_aggregated_features[1,1] - A[1,0] * Conv_0_Relu_ts[0,1]
   - A[1,1] * Conv_0_Relu_ts[1,1]
   + A[1,1] * Conv_1_aggregated_features[1,1]
   - A[1,2] * Conv_0_Relu_ts[2,1] - A[1,3] * Conv_0_Relu_ts[3,1]
   - A[1,4] * Conv_0_Relu_ts[4,1]
   + A[2,1] * Conv_1_aggregated_features[1,1]
   + A[3,1] * Conv_1_aggregated_features[1,1]
   + A[4,1] * Conv_1_aggregated_features[1,1] ] = 0
 Conv_1_averages_constraint[1,2]: [
   A[0,1] * Conv_1_aggregated_features[1,2] - A[1,0] * Conv_0_Relu_ts[0,2]
   - A[1,1] * Conv_0_Relu_ts[1,2]
   + A[1,1] * Conv_1_aggregated_features[1,2]
   - A[1,2] * Conv_0_Relu_ts[2,2] - A[1,3] * Conv_0_Relu_ts[3,2]
   - A[1,4] * Conv_0_Relu_ts[4,2]
   + A[2,1] * Conv_1_aggregated_features[1,2]
   + A[3,1] * Conv_1_aggregated_features[1,2]
   + A[4,1] * Conv_1_aggregated_features[1,2] ] = 0
 Conv_1_averages_constraint[1,3]: [
   A[0,1] * Conv_1_aggregated_features[1,3] - A[1,0] * Conv_0_Relu_ts[0,3]
   - A[1,1] * Conv_0_Relu_ts[1,3]
   + A[1,1] * Conv_1_aggregated_features[1,3]
   - A[1,2] * Conv_0_Relu_ts[2,3] - A[1,3] * Conv_0_Relu_ts[3,3]
   - A[1,4] * Conv_0_Relu_ts[4,3]
   + A[2,1] * Conv_1_aggregated_features[1,3]
   + A[3,1] * Conv_1_aggregated_features[1,3]
   + A[4,1] * Conv_1_aggregated_features[1,3] ] = 0
 Conv_1_averages_constraint[2,0]: [
   A[0,2] * Conv_1_aggregated_features[2,0]
   + A[1,2] * Conv_1_aggregated_features[2,0]
   - A[2,0] * Conv_0_Relu_ts[0,0] - A[2,1] * Conv_0_Relu_ts[1,0]
   - A[2,2] * Conv_0_Relu_ts[2,0]
   + A[2,2] * Conv_1_aggregated_features[2,0]
   - A[2,3] * Conv_0_Relu_ts[3,0] - A[2,4] * Conv_0_Relu_ts[4,0]
   + A[3,2] * Conv_1_aggregated_features[2,0]
   + A[4,2] * Conv_1_aggregated_features[2,0] ] = 0
 Conv_1_averages_constraint[2,1]: [
   A[0,2] * Conv_1_aggregated_features[2,1]
   + A[1,2] * Conv_1_aggregated_features[2,1]
   - A[2,0] * Conv_0_Relu_ts[0,1] - A[2,1] * Conv_0_Relu_ts[1,1]
   - A[2,2] * Conv_0_Relu_ts[2,1]
   + A[2,2] * Conv_1_aggregated_features[2,1]
   - A[2,3] * Conv_0_Relu_ts[3,1] - A[2,4] * Conv_0_Relu_ts[4,1]
   + A[3,2] * Conv_1_aggregated_features[2,1]
   + A[4,2] * Conv_1_aggregated_features[2,1] ] = 0
 Conv_1_averages_constraint[2,2]: [
   A[0,2] * Conv_1_aggregated_features[2,2]
   + A[1,2] * Conv_1_aggregated_features[2,2]
   - A[2,0] * Conv_0_Relu_ts[0,2] - A[2,1] * Conv_0_Relu_ts[1,2]
   - A[2,2] * Conv_0_Relu_ts[2,2]
   + A[2,2] * Conv_1_aggregated_features[2,2]
   - A[2,3] * Conv_0_Relu_ts[3,2] - A[2,4] * Conv_0_Relu_ts[4,2]
   + A[3,2] * Conv_1_aggregated_features[2,2]
   + A[4,2] * Conv_1_aggregated_features[2,2] ] = 0
 Conv_1_averages_constraint[2,3]: [
   A[0,2] * Conv_1_aggregated_features[2,3]
   + A[1,2] * Conv_1_aggregated_features[2,3]
   - A[2,0] * Conv_0_Relu_ts[0,3] - A[2,1] * Conv_0_Relu_ts[1,3]
   - A[2,2] * Conv_0_Relu_ts[2,3]
   + A[2,2] * Conv_1_aggregated_features[2,3]
   - A[2,3] * Conv_0_Relu_ts[3,3] - A[2,4] * Conv_0_Relu_ts[4,3]
   + A[3,2] * Conv_1_aggregated_features[2,3]
   + A[4,2] * Conv_1_aggregated_features[2,3] ] = 0
 Conv_1_averages_constraint[3,0]: [
   A[0,3] * Conv_1_aggregated_features[3,0]
   + A[1,3] * Conv_1_aggregated_features[3,0]
   + A[2,3] * Conv_1_aggregated_features[3,0]
   - A[3,0] * Conv_0_Relu_ts[0,0] - A[3,1] * Conv_0_Relu_ts[1,0]
   - A[3,2] * Conv_0_Relu_ts[2,0] - A[3,3] * Conv_0_Relu_ts[3,0]
   + A[3,3] * Conv_1_aggregated_features[3,0]
   - A[3,4] * Conv_0_Relu_ts[4,0]
   + A[4,3] * Conv_1_aggregated_features[3,0] ] = 0
 Conv_1_averages_constraint[3,1]: [
   A[0,3] * Conv_1_aggregated_features[3,1]
   + A[1,3] * Conv_1_aggregated_features[3,1]
   + A[2,3] * Conv_1_aggregated_features[3,1]
   - A[3,0] * Conv_0_Relu_ts[0,1] - A[3,1] * Conv_0_Relu_ts[1,1]
   - A[3,2] * Conv_0_Relu_ts[2,1] - A[3,3] * Conv_0_Relu_ts[3,1]
   + A[3,3] * Conv_1_aggregated_features[3,1]
   - A[3,4] * Conv_0_Relu_ts[4,1]
   + A[4,3] * Conv_1_aggregated_features[3,1] ] = 0
 Conv_1_averages_constraint[3,2]: [
   A[0,3] * Conv_1_aggregated_features[3,2]
   + A[1,3] * Conv_1_aggregated_features[3,2]
   + A[2,3] * Conv_1_aggregated_features[3,2]
   - A[3,0] * Conv_0_Relu_ts[0,2] - A[3,1] * Conv_0_Relu_ts[1,2]
   - A[3,2] * Conv_0_Relu_ts[2,2] - A[3,3] * Conv_0_Relu_ts[3,2]
   + A[3,3] * Conv_1_aggregated_features[3,2]
   - A[3,4] * Conv_0_Relu_ts[4,2]
   + A[4,3] * Conv_1_aggregated_features[3,2] ] = 0
 Conv_1_averages_constraint[3,3]: [
   A[0,3] * Conv_1_aggregated_features[3,3]
   + A[1,3] * Conv_1_aggregated_features[3,3]
   + A[2,3] * Conv_1_aggregated_features[3,3]
   - A[3,0] * Conv_0_Relu_ts[0,3] - A[3,1] * Conv_0_Relu_ts[1,3]
   - A[3,2] * Conv_0_Relu_ts[2,3] - A[3,3] * Conv_0_Relu_ts[3,3]
   + A[3,3] * Conv_1_aggregated_features[3,3]
   - A[3,4] * Conv_0_Relu_ts[4,3]
   + A[4,3] * Conv_1_aggregated_features[3,3] ] = 0
 Conv_1_averages_constraint[4,0]: [
   A[0,4] * Conv_1_aggregated_features[4,0]
   + A[1,4] * Conv_1_aggregated_features[4,0]
   + A[2,4] * Conv_1_aggregated_features[4,0]
   + A[3,4] * Conv_1_aggregated_features[4,0]
   - A[4,0] * Conv_0_Relu_ts[0,0] - A[4,1] * Conv_0_Relu_ts[1,0]
   - A[4,2] * Conv_0_Relu_ts[2,0] - A[4,3] * Conv_0_Relu_ts[3,0]
   - A[4,4] * Conv_0_Relu_ts[4,0]
   + A[4,4] * Conv_1_aggregated_features[4,0] ] = 0
 Conv_1_averages_constraint[4,1]: [
   A[0,4] * Conv_1_aggregated_features[4,1]
   + A[1,4] * Conv_1_aggregated_features[4,1]
   + A[2,4] * Conv_1_aggregated_features[4,1]
   + A[3,4] * Conv_1_aggregated_features[4,1]
   - A[4,0] * Conv_0_Relu_ts[0,1] - A[4,1] * Conv_0_Relu_ts[1,1]
   - A[4,2] * Conv_0_Relu_ts[2,1] - A[4,3] * Conv_0_Relu_ts[3,1]
   - A[4,4] * Conv_0_Relu_ts[4,1]
   + A[4,4] * Conv_1_aggregated_features[4,1] ] = 0
 Conv_1_averages_constraint[4,2]: [
   A[0,4] * Conv_1_aggregated_features[4,2]
   + A[1,4] * Conv_1_aggregated_features[4,2]
   + A[2,4] * Conv_1_aggregated_features[4,2]
   + A[3,4] * Conv_1_aggregated_features[4,2]
   - A[4,0] * Conv_0_Relu_ts[0,2] - A[4,1] * Conv_0_Relu_ts[1,2]
   - A[4,2] * Conv_0_Relu_ts[2,2] - A[4,3] * Conv_0_Relu_ts[3,2]
   - A[4,4] * Conv_0_Relu_ts[4,2]
   + A[4,4] * Conv_1_aggregated_features[4,2] ] = 0
 Conv_1_averages_constraint[4,3]: [
   A[0,4] * Conv_1_aggregated_features[4,3]
   + A[1,4] * Conv_1_aggregated_features[4,3]
   + A[2,4] * Conv_1_aggregated_features[4,3]
   + A[3,4] * Conv_1_aggregated_features[4,3]
   - A[4,0] * Conv_0_Relu_ts[0,3] - A[4,1] * Conv_0_Relu_ts[1,3]
   - A[4,2] * Conv_0_Relu_ts[2,3] - A[4,3] * Conv_0_Relu_ts[3,3]
   - A[4,4] * Conv_0_Relu_ts[4,3]
   + A[4,4] * Conv_1_aggregated_features[4,3] ] = 0
 cosine_similarity: 15.11257266998291 Aggregation[0]
   + 0.2722989022731781 Aggregation[1] + 1.393101096153259 Aggregation[2]
   + 8.027902603149414 Aggregation[3] + [
   - 17.1712532043457 embedding_similarity * embedding_magnitude ] = 0
Bounds
 X[0,0] <= 5
 X[1,0] <= 5
 X[2,0] <= 5
 X[3,0] <= 5
 X[4,0] <= 5
 Conv_0_aggregated_features[0,0] <= 5
 Conv_0_aggregated_features[1,0] <= 5
 Conv_0_aggregated_features[2,0] <= 5
 Conv_0_aggregated_features[3,0] <= 5
 Conv_0_aggregated_features[4,0] <= 5
 -5.619521796703339 <= Conv_0_t[0,0] <= 8.299143254756927
 -2.621069669723511 <= Conv_0_t[0,1] <= 0.4476426839828491
 -7.29360830783844 <= Conv_0_t[0,2] <= -0.372223973274231
 -2.056674525141716 <= Conv_0_t[0,3] <= 5.947406053543091
 -5.619521796703339 <= Conv_0_t[1,0] <= 8.299143254756927
 -2.621069669723511 <= Conv_0_t[1,1] <= 0.4476426839828491
 -7.29360830783844 <= Conv_0_t[1,2] <= -0.372223973274231
 -2.056674525141716 <= Conv_0_t[1,3] <= 5.947406053543091
 -5.619521796703339 <= Conv_0_t[2,0] <= 8.299143254756927
 -2.621069669723511 <= Conv_0_t[2,1] <= 0.4476426839828491
 -7.29360830783844 <= Conv_0_t[2,2] <= -0.372223973274231
 -2.056674525141716 <= Conv_0_t[2,3] <= 5.947406053543091
 -5.619521796703339 <= Conv_0_t[3,0] <= 8.299143254756927
 -2.621069669723511 <= Conv_0_t[3,1] <= 0.4476426839828491
 -7.29360830783844 <= Conv_0_t[3,2] <= -0.372223973274231
 -2.056674525141716 <= Conv_0_t[3,3] <= 5.947406053543091
 -5.619521796703339 <= Conv_0_t[4,0] <= 8.299143254756927
 -2.621069669723511 <= Conv_0_t[4,1] <= 0.4476426839828491
 -7.29360830783844 <= Conv_0_t[4,2] <= -0.372223973274231
 -2.056674525141716 <= Conv_0_t[4,3] <= 5.947406053543091
 Conv_0_Relu_ts[0,0] <= 8.299143254756927
 Conv_0_Relu_ts[0,1] <= 0.4476426839828491
 Conv_0_Relu_ts[0,2] = 0
 Conv_0_Relu_ts[0,3] <= 5.947406053543091
 Conv_0_Relu_ts[1,0] <= 8.299143254756927
 Conv_0_Relu_ts[1,1] <= 0.4476426839828491
 Conv_0_Relu_ts[1,2] = 0
 Conv_0_Relu_ts[1,3] <= 5.947406053543091
 Conv_0_Relu_ts[2,0] <= 8.299143254756927
 Conv_0_Relu_ts[2,1] <= 0.4476426839828491
 Conv_0_Relu_ts[2,2] = 0
 Conv_0_Relu_ts[2,3] <= 5.947406053543091
 Conv_0_Relu_ts[3,0] <= 8.299143254756927
 Conv_0_Relu_ts[3,1] <= 0.4476426839828491
 Conv_0_Relu_ts[3,2] = 0
 Conv_0_Relu_ts[3,3] <= 5.947406053543091
 Conv_0_Relu_ts[4,0] <= 8.299143254756927
 Conv_0_Relu_ts[4,1] <= 0.4476426839828491
 Conv_0_Relu_ts[4,2] = 0
 Conv_0_Relu_ts[4,3] <= 5.947406053543091
 Conv_0_Relu_ss[0,0] <= 5.619521796703339
 Conv_0_Relu_ss[0,1] <= 2.621069669723511
 Conv_0_Relu_ss[0,2] <= 7.29360830783844
 Conv_0_Relu_ss[0,3] <= 2.056674525141716
 Conv_0_Relu_ss[1,0] <= 5.619521796703339
 Conv_0_Relu_ss[1,1] <= 2.621069669723511
 Conv_0_Relu_ss[1,2] <= 7.29360830783844
 Conv_0_Relu_ss[1,3] <= 2.056674525141716
 Conv_0_Relu_ss[2,0] <= 5.619521796703339
 Conv_0_Relu_ss[2,1] <= 2.621069669723511
 Conv_0_Relu_ss[2,2] <= 7.29360830783844
 Conv_0_Relu_ss[2,3] <= 2.056674525141716
 Conv_0_Relu_ss[3,0] <= 5.619521796703339
 Conv_0_Relu_ss[3,1] <= 2.621069669723511
 Conv_0_Relu_ss[3,2] <= 7.29360830783844
 Conv_0_Relu_ss[3,3] <= 2.056674525141716
 Conv_0_Relu_ss[4,0] <= 5.619521796703339
 Conv_0_Relu_ss[4,1] <= 2.621069669723511
 Conv_0_Relu_ss[4,2] <= 7.29360830783844
 Conv_0_Relu_ss[4,3] <= 2.056674525141716
 Conv_1_aggregated_features[0,0] <= 8.299143254756927
 Conv_1_aggregated_features[0,1] <= 0.4476426839828491
 Conv_1_aggregated_features[0,2] = 0
 Conv_1_aggregated_features[0,3] <= 5.947406053543091
 Conv_1_aggregated_features[1,0] <= 8.299143254756927
 Conv_1_aggregated_features[1,1] <= 0.4476426839828491
 Conv_1_aggregated_features[1,2] = 0
 Conv_1_aggregated_features[1,3] <= 5.947406053543091
 Conv_1_aggregated_features[2,0] <= 8.299143254756927
 Conv_1_aggregated_features[2,1] <= 0.4476426839828491
 Conv_1_aggregated_features[2,2] = 0
 Conv_1_aggregated_features[2,3] <= 5.947406053543091
 Conv_1_aggregated_features[3,0] <= 8.299143254756927
 Conv_1_aggregated_features[3,1] <= 0.4476426839828491
 Conv_1_aggregated_features[3,2] = 0
 Conv_1_aggregated_features[3,3] <= 5.947406053543091
 Conv_1_aggregated_features[4,0] <= 8.299143254756927
 Conv_1_aggregated_features[4,1] <= 0.4476426839828491
 Conv_1_aggregated_features[4,2] = 0
 Conv_1_aggregated_features[4,3] <= 5.947406053543091
 -17.44104739088009 <= Conv_1_t[0,0] <= 9.626689280521735
 -7.54207430137968 <= Conv_1_t[0,1] <= 3.840047017460973
 -17.89046704265052 <= Conv_1_t[0,2] <= 13.3664995518295
 -1.181242947862494 <= Conv_1_t[0,3] <= 19.80780815186422
 -17.44104739088009 <= Conv_1_t[1,0] <= 9.626689280521735
 -7.54207430137968 <= Conv_1_t[1,1] <= 3.840047017460973
 -17.89046704265052 <= Conv_1_t[1,2] <= 13.3664995518295
 -1.181242947862494 <= Conv_1_t[1,3] <= 19.80780815186422
 -17.44104739088009 <= Conv_1_t[2,0] <= 9.626689280521735
 -7.54207430137968 <= Conv_1_t[2,1] <= 3.840047017460973
 -17.89046704265052 <= Conv_1_t[2,2] <= 13.3664995518295
 -1.181242947862494 <= Conv_1_t[2,3] <= 19.80780815186422
 -17.44104739088009 <= Conv_1_t[3,0] <= 9.626689280521735
 -7.54207430137968 <= Conv_1_t[3,1] <= 3.840047017460973
 -17.89046704265052 <= Conv_1_t[3,2] <= 13.3664995518295
 -1.181242947862494 <= Conv_1_t[3,3] <= 19.80780815186422
 -17.44104739088009 <= Conv_1_t[4,0] <= 9.626689280521735
 -7.54207430137968 <= Conv_1_t[4,1] <= 3.840047017460973
 -17.89046704265052 <= Conv_1_t[4,2] <= 13.3664995518295
 -1.181242947862494 <= Conv_1_t[4,3] <= 19.80780815186422
 Conv_1_Relu_ts[0,0] <= 9.626689280521735
 Conv_1_Relu_ts[0,1] <= 3.840047017460973
 Conv_1_Relu_ts[0,2] <= 13.3664995518295
 Conv_1_Relu_ts[0,3] <= 19.80780815186422
 Conv_1_Relu_ts[1,0] <= 9.626689280521735
 Conv_1_Relu_ts[1,1] <= 3.840047017460973
 Conv_1_Relu_ts[1,2] <= 13.3664995518295
 Conv_1_Relu_ts[1,3] <= 19.80780815186422
 Conv_1_Relu_ts[2,0] <= 9.626689280521735
 Conv_1_Relu_ts[2,1] <= 3.840047017460973
 Conv_1_Relu_ts[2,2] <= 13.3664995518295
 Conv_1_Relu_ts[2,3] <= 19.80780815186422
 Conv_1_Relu_ts[3,0] <= 9.626689280521735
 Conv_1_Relu_ts[3,1] <= 3.840047017460973
 Conv_1_Relu_ts[3,2] <= 13.3664995518295
 Conv_1_Relu_ts[3,3] <= 19.80780815186422
 Conv_1_Relu_ts[4,0] <= 9.626689280521735
 Conv_1_Relu_ts[4,1] <= 3.840047017460973
 Conv_1_Relu_ts[4,2] <= 13.3664995518295
 Conv_1_Relu_ts[4,3] <= 19.80780815186422
 Conv_1_Relu_ss[0,0] <= 17.44104739088009
 Conv_1_Relu_ss[0,1] <= 7.54207430137968
 Conv_1_Relu_ss[0,2] <= 17.89046704265052
 Conv_1_Relu_ss[0,3] <= 1.181242947862494
 Conv_1_Relu_ss[1,0] <= 17.44104739088009
 Conv_1_Relu_ss[1,1] <= 7.54207430137968
 Conv_1_Relu_ss[1,2] <= 17.89046704265052
 Conv_1_Relu_ss[1,3] <= 1.181242947862494
 Conv_1_Relu_ss[2,0] <= 17.44104739088009
 Conv_1_Relu_ss[2,1] <= 7.54207430137968
 Conv_1_Relu_ss[2,2] <= 17.89046704265052
 Conv_1_Relu_ss[2,3] <= 1.181242947862494
 Conv_1_Relu_ss[3,0] <= 17.44104739088009
 Conv_1_Relu_ss[3,1] <= 7.54207430137968
 Conv_1_Relu_ss[3,2] <= 17.89046704265052
 Conv_1_Relu_ss[3,3] <= 1.181242947862494
 Conv_1_Relu_ss[4,0] <= 17.44104739088009
 Conv_1_Relu_ss[4,1] <= 7.54207430137968
 Conv_1_Relu_ss[4,2] <= 17.89046704265052
 Conv_1_Relu_ss[4,3] <= 1.181242947862494
 Aggregation[0] <= 9.626689280521735
 Aggregation[1] <= 3.840047017460974
 Aggregation[2] <= 13.3664995518295
 Aggregation[3] <= 19.80780815186422
 -108.3558361054767 <= Lin_0_t[0] <= 19.51431114711157
 -12.29928911947451 <= Lin_0_t[1] <= 4.396936156963867
 -10.60863422515173 <= Lin_0_t[2] <= 58.84170178995957
 -14.84236966431347 <= Lin_0_t[3] <= 0.430299686915956
 Lin_0_Relu_ts[0,0] <= 19.51431114711157
 Lin_0_Relu_ts[0,1] <= 4.396936156963867
 Lin_0_Relu_ts[0,2] <= 58.84170178995957
 Lin_0_Relu_ts[0,3] <= 0.430299686915956
 Lin_0_Relu_ss[0,0] <= 108.3558361054767
 Lin_0_Relu_ss[0,1] <= 12.29928911947451
 Lin_0_Relu_ss[0,2] <= 10.60863422515173
 Lin_0_Relu_ss[0,3] <= 14.84236966431347
 -32.45466270347528 <= Output_t[0] <= 112.1958014674212
 embedding_similarity <= 1
Binaries
 A[0,0] A[0,1] A[0,2] A[0,3] A[0,4] A[1,0] A[1,1] A[1,2] A[1,3] A[1,4]
 A[2,0] A[2,1] A[2,2] A[2,3] A[2,4] A[3,0] A[3,1] A[3,2] A[3,3] A[3,4]
 A[4,0] A[4,1] A[4,2] A[4,3] A[4,4] C95 C96 C97 C98 C99 C100 C101 C102 C103
 C104 C105 C106 C107 C108 C109 C110 C111 C112 C113 C114 C195 C196 C197 C198
 C199 C200 C201 C202 C203 C204 C205 C206 C207 C208 C209 C210 C211 C212 C213
 C214 C231 C232 C233 C234
Generals
 X[0,0] X[1,0] X[2,0] X[3,0] X[4,0]
General Constraints
 embedding_magnitude_constraint: embedding_magnitude = NORM ( 2 ) (
   Aggregation[0] , Aggregation[1] , Aggregation[2] , Aggregation[3] )
End
