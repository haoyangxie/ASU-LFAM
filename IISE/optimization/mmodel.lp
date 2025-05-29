\ Model speed_control
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 240 temperature0 - 240 temperature1 - 240 temperature2
   - 240 temperature3 - 240 temperature4 - 240 temperature5
   - 240 temperature6 - 240 temperature7 - 240 temperature8
   - 240 temperature9 - 240 temperature10 - 240 temperature11
   - 240 temperature12 - 240 temperature13 - 240 temperature14
   - 240 temperature15 - 240 temperature16 - 240 temperature17
   - 240 temperature18 - 240 temperature19 - 240 temperature20
   - 240 temperature21 - 240 temperature22 - 240 temperature23
   - 240 temperature24 - 240 temperature25 - 240 temperature26
   - 240 temperature27 - 240 temperature28 - 240 temperature29
   - 240 temperature30 - 240 temperature31 - 240 temperature32
   - 240 temperature33 - 240 temperature34 - 240 temperature35
   - 240 temperature36 - 240 temperature37 - 240 temperature38
   - 240 temperature39 - 240 temperature40 - 240 temperature41
   - 240 temperature42 - 240 temperature43 - 240 temperature44
   - 240 temperature45 - 240 temperature46 - 240 temperature47
   - 240 temperature48 - 240 temperature49 - 240 temperature50
   - 240 temperature51 - 240 temperature52 - 240 temperature53
   + 777600 Constant + [ 2 temperature0 ^2 + 2 temperature1 ^2
   + 2 temperature2 ^2 + 2 temperature3 ^2 + 2 temperature4 ^2
   + 2 temperature5 ^2 + 2 temperature6 ^2 + 2 temperature7 ^2
   + 2 temperature8 ^2 + 2 temperature9 ^2 + 2 temperature10 ^2
   + 2 temperature11 ^2 + 2 temperature12 ^2 + 2 temperature13 ^2
   + 2 temperature14 ^2 + 2 temperature15 ^2 + 2 temperature16 ^2
   + 2 temperature17 ^2 + 2 temperature18 ^2 + 2 temperature19 ^2
   + 2 temperature20 ^2 + 2 temperature21 ^2 + 2 temperature22 ^2
   + 2 temperature23 ^2 + 2 temperature24 ^2 + 2 temperature25 ^2
   + 2 temperature26 ^2 + 2 temperature27 ^2 + 2 temperature28 ^2
   + 2 temperature29 ^2 + 2 temperature30 ^2 + 2 temperature31 ^2
   + 2 temperature32 ^2 + 2 temperature33 ^2 + 2 temperature34 ^2
   + 2 temperature35 ^2 + 2 temperature36 ^2 + 2 temperature37 ^2
   + 2 temperature38 ^2 + 2 temperature39 ^2 + 2 temperature40 ^2
   + 2 temperature41 ^2 + 2 temperature42 ^2 + 2 temperature43 ^2
   + 2 temperature44 ^2 + 2 temperature45 ^2 + 2 temperature46 ^2
   + 2 temperature47 ^2 + 2 temperature48 ^2 + 2 temperature49 ^2
   + 2 temperature50 ^2 + 2 temperature51 ^2 + 2 temperature52 ^2
   + 2 temperature53 ^2 ] / 2 
Subject To
 constr_v0_lower: v0 - 0.2 z0 >= 0
 constr_v0_upper: v0 - 0.2 z0 <= 0
 R2: t0 - 66 y0 = 174.9
 R3: t1 - 66 y0 - 66 y1 = 171.6
 R4: t2 - 66 y0 - 66 y1 - 66 y2 = 168.3
 R5: t3 - 66 y0 - 66 y1 - 66 y2 - 66 y3 = 165
 R6: t4 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 = 161.7
 R7: t5 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 = 158.4
 R8: t6 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 = 155.1
 R9: t7 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   = 151.8
 R10: t8 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 = 148.5
 R11: t9 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 = 145.2
 R12: t10 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 = 141.9
 R13: t11 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 = 138.6
 R14: t12 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 = 135.3
 R15: t13 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 = 132
 R16: t14 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 = 128.7
 R17: t15 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   = 125.4
 R18: t16 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 = 122.1
 R19: t17 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 = 118.8
 R20: t18 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 = 115.5
 R21: t19 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 = 112.2
 R22: t20 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 = 108.9
 R23: t21 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 = 105.6
 R24: t22 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 = 102.3
 R25: t23 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   = 99
 R26: t24 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 = 95.7
 R27: t25 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 = 92.4
 R28: t26 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 = 89.1
 R29: t27 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 = 85.8
 R30: t28 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 = 82.5
 R31: t29 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 = 79.2
 R32: t30 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 = 75.9
 R33: t31 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   = 72.6
 R34: t32 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 = 69.3
 R35: t33 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 = 66
 R36: t34 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 = 62.7
 R37: t35 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 = 59.4
 R38: t36 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 = 56.1
 R39: t37 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 = 52.8
 R40: t38 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 = 49.5
 R41: t39 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   = 46.2
 R42: t40 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 = 42.9
 R43: t41 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 = 39.6
 R44: t42 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 = 36.3
 R45: t43 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 = 33
 R46: t44 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 = 29.7
 R47: t45 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 = 26.4
 R48: t46 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 = 23.1
 R49: t47 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 - 66 y47
   = 19.8
 R50: t48 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 - 66 y47
   - 66 y48 = 16.5
 R51: t49 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 - 66 y47
   - 66 y48 - 66 y49 = 13.2
 R52: t50 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 - 66 y47
   - 66 y48 - 66 y49 - 66 y50 = 9.9
 R53: t51 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 - 66 y47
   - 66 y48 - 66 y49 - 66 y50 - 66 y51 = 6.6
 R54: t52 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 - 66 y47
   - 66 y48 - 66 y49 - 66 y50 - 66 y51 - 66 y52 = 3.3
 R55: t53 - 66 y0 - 66 y1 - 66 y2 - 66 y3 - 66 y4 - 66 y5 - 66 y6 - 66 y7
   - 66 y8 - 66 y9 - 66 y10 - 66 y11 - 66 y12 - 66 y13 - 66 y14 - 66 y15
   - 66 y16 - 66 y17 - 66 y18 - 66 y19 - 66 y20 - 66 y21 - 66 y22 - 66 y23
   - 66 y24 - 66 y25 - 66 y26 - 66 y27 - 66 y28 - 66 y29 - 66 y30 - 66 y31
   - 66 y32 - 66 y33 - 66 y34 - 66 y35 - 66 y36 - 66 y37 - 66 y38 - 66 y39
   - 66 y40 - 66 y41 - 66 y42 - 66 y43 - 66 y44 - 66 y45 - 66 y46 - 66 y47
   - 66 y48 - 66 y49 - 66 y50 - 66 y51 - 66 y52 - 66 y53 = 0
 mul_constr0: 0.0116635708827124 t0 + mul_var0 = 0
 temperature_constr0: - 105.831091297194 exp_term0 + temperature0
   = 97.16890870280604
 temp_lower_0: temperature0 >= 90
 temp_upper_0: temperature0 <= 150
 mul_constr1: 0.0104088520504559 t1 + mul_var1 = 0
 temperature_constr1: - 106.5399314807765 exp_term1 + temperature1
   = 95.4600685192235
 temp_lower_1: temperature1 >= 90
 temp_upper_1: temperature1 <= 150
 mul_constr2: 0.00987724133725384 t2 + mul_var2 = 0
 temperature_constr2: - 110.6099623918955 exp_term2 + temperature2
   = 94.39003760810451
 temp_lower_2: temperature2 >= 90
 temp_upper_2: temperature2 <= 150
 mul_constr3: 0.0109573545220085 t3 + mul_var3 = 0
 temperature_constr3: - 109.1321661058863 exp_term3 + temperature3
   = 96.86783389411372
 temp_lower_3: temperature3 >= 90
 temp_upper_3: temperature3 <= 150
 mul_constr4: 0.0111583876560147 t4 + mul_var4 = 0
 temperature_constr4: - 109.1333424959949 exp_term4 + temperature4
   = 96.86665750400509
 temp_lower_4: temperature4 >= 90
 temp_upper_4: temperature4 <= 150
 mul_constr5: 0.0105547925988143 t5 + mul_var5 = 0
 temperature_constr5: - 109.0709088934738 exp_term5 + temperature5
   = 95.92909110652623
 temp_lower_5: temperature5 >= 90
 temp_upper_5: temperature5 <= 150
 mul_constr6: 0.0102138412534222 t6 + mul_var6 = 0
 temperature_constr6: - 111.0948037358883 exp_term6 + temperature6
   = 93.90519626411174
 temp_lower_6: temperature6 >= 90
 temp_upper_6: temperature6 <= 150
 mul_constr7: 0.0119275355706226 t7 + mul_var7 = 0
 temperature_constr7: - 108.4522872229003 exp_term7 + temperature7
   = 98.54771277709969
 temp_lower_7: temperature7 >= 90
 temp_upper_7: temperature7 <= 150
 mul_constr8: 0.0110573821265464 t8 + mul_var8 = 0
 temperature_constr8: - 110.7241220145337 exp_term8 + temperature8
   = 95.27587798546629
 temp_lower_8: temperature8 >= 90
 temp_upper_8: temperature8 <= 150
 mul_constr9: 0.00955513865780049 t9 + mul_var9 = 0
 temperature_constr9: - 115.9338918998374 exp_term9 + temperature9
   = 91.06610810016257
 temp_lower_9: temperature9 >= 90
 temp_upper_9: temperature9 <= 150
 mul_constr10: 0.00958054103605975 t10 + mul_var10 = 0
 temperature_constr10: - 117.6248546611039 exp_term10 + temperature10
   = 89.37514533889609
 temp_lower_10: temperature10 >= 90
 temp_upper_10: temperature10 <= 150
 mul_constr11: 0.0092335593933891 t11 + mul_var11 = 0
 temperature_constr11: - 123.4074412275519 exp_term11 + temperature11
   = 84.59255877244809
 temp_lower_11: temperature11 >= 90
 temp_upper_11: temperature11 <= 150
 mul_constr12: 0.0100209801065354 t12 + mul_var12 = 0
 temperature_constr12: - 115.9010419190696 exp_term12 + temperature12
   = 91.09895808093042
 temp_lower_12: temperature12 >= 90
 temp_upper_12: temperature12 <= 150
 mul_constr13: 0.0105051632041596 t13 + mul_var13 = 0
 temperature_constr13: - 112.0732669547247 exp_term13 + temperature13
   = 93.92673304527526
 temp_lower_13: temperature13 >= 90
 temp_upper_13: temperature13 <= 150
 mul_constr14: 0.00935712031425188 t14 + mul_var14 = 0
 temperature_constr14: - 124.2747780956032 exp_term14 + temperature14
   = 82.72522190439682
 temp_lower_14: temperature14 >= 90
 temp_upper_14: temperature14 <= 150
 mul_constr15: 0.0121229409569728 t15 + mul_var15 = 0
 temperature_constr15: - 107.1543031891656 exp_term15 + temperature15
   = 99.84569681083435
 temp_lower_15: temperature15 >= 90
 temp_upper_15: temperature15 <= 150
 mul_constr16: 0.0128669309492976 t16 + mul_var16 = 0
 temperature_constr16: - 103.5858355686346 exp_term16 + temperature16
   = 101.4141644313654
 temp_lower_16: temperature16 >= 90
 temp_upper_16: temperature16 <= 150
 mul_constr17: 0.0135515623747671 t17 + mul_var17 = 0
 temperature_constr17: - 101.4035973448839 exp_term17 + temperature17
   = 103.5964026551161
 temp_lower_17: temperature17 >= 90
 temp_upper_17: temperature17 <= 150
 mul_constr18: 0.0153648814794203 t18 + mul_var18 = 0
 temperature_constr18: - 98.40436003188577 exp_term18 + temperature18
   = 107.5956399681142
 temp_lower_18: temperature18 >= 90
 temp_upper_18: temperature18 <= 150
 mul_constr19: 0.0167728341471456 t19 + mul_var19 = 0
 temperature_constr19: - 91.31712847210329 exp_term19 + temperature19
   = 113.6828715278967
 temp_lower_19: temperature19 >= 90
 temp_upper_19: temperature19 <= 150
 mul_constr20: 0.0166693370230336 t20 + mul_var20 = 0
 temperature_constr20: - 91.15700758143295 exp_term20 + temperature20
   = 112.8429924185671
 temp_lower_20: temperature20 >= 90
 temp_upper_20: temperature20 <= 150
 mul_constr21: 0.0170896155716006 t21 + mul_var21 = 0
 temperature_constr21: - 87.44861450934243 exp_term21 + temperature21
   = 114.5513854906576
 temp_lower_21: temperature21 >= 90
 temp_upper_21: temperature21 <= 150
 mul_constr22: 0.0161053049294895 t22 + mul_var22 = 0
 temperature_constr22: - 94.67132577072967 exp_term22 + temperature22
   = 109.3286742292703
 temp_lower_22: temperature22 >= 90
 temp_upper_22: temperature22 <= 150
 mul_constr23: 0.0171034378137564 t23 + mul_var23 = 0
 temperature_constr23: - 91.69803084683289 exp_term23 + temperature23
   = 113.3019691531671
 temp_lower_23: temperature23 >= 90
 temp_upper_23: temperature23 <= 150
 mul_constr24: 0.0168033884855486 t24 + mul_var24 = 0
 temperature_constr24: - 92.47595969901796 exp_term24 + temperature24
   = 112.524040300982
 temp_lower_24: temperature24 >= 90
 temp_upper_24: temperature24 <= 150
 mul_constr25: 0.0179760322113699 t25 + mul_var25 = 0
 temperature_constr25: - 90.06718170290705 exp_term25 + temperature25
   = 113.932818297093
 temp_lower_25: temperature25 >= 90
 temp_upper_25: temperature25 <= 150
 mul_constr26: 0.0186322174339244 t26 + mul_var26 = 0
 temperature_constr26: - 81.371956441943 exp_term26 + temperature26
   = 118.628043558057
 temp_lower_26: temperature26 >= 90
 temp_upper_26: temperature26 <= 150
 mul_constr27: 0.0105547925988143 t27 + mul_var27 = 0
 temperature_constr27: - 109.0709088934738 exp_term27 + temperature27
   = 95.92909110652623
 temp_lower_27: temperature27 >= 90
 temp_upper_27: temperature27 <= 150
 mul_constr28: 0.0110573821265464 t28 + mul_var28 = 0
 temperature_constr28: - 110.7241220145337 exp_term28 + temperature28
   = 95.27587798546629
 temp_lower_28: temperature28 >= 90
 temp_upper_28: temperature28 <= 150
 mul_constr29: 0.0104088520504559 t29 + mul_var29 = 0
 temperature_constr29: - 106.5399314807765 exp_term29 + temperature29
   = 95.4600685192235
 temp_lower_29: temperature29 >= 90
 temp_upper_29: temperature29 <= 150
 mul_constr30: 0.0109573545220085 t30 + mul_var30 = 0
 temperature_constr30: - 109.1321661058863 exp_term30 + temperature30
   = 96.86783389411372
 temp_lower_30: temperature30 >= 90
 temp_upper_30: temperature30 <= 150
 mul_constr31: 0.0111583876560147 t31 + mul_var31 = 0
 temperature_constr31: - 109.1333424959949 exp_term31 + temperature31
   = 96.86665750400509
 temp_lower_31: temperature31 >= 90
 temp_upper_31: temperature31 <= 150
 mul_constr32: 0.00987724133725384 t32 + mul_var32 = 0
 temperature_constr32: - 110.6099623918955 exp_term32 + temperature32
   = 94.39003760810451
 temp_lower_32: temperature32 >= 90
 temp_upper_32: temperature32 <= 150
 mul_constr33: 0.0119275355706226 t33 + mul_var33 = 0
 temperature_constr33: - 108.4522872229003 exp_term33 + temperature33
   = 98.54771277709969
 temp_lower_33: temperature33 >= 90
 temp_upper_33: temperature33 <= 150
 mul_constr34: 0.0102138412534222 t34 + mul_var34 = 0
 temperature_constr34: - 111.0948037358883 exp_term34 + temperature34
   = 93.90519626411174
 temp_lower_34: temperature34 >= 90
 temp_upper_34: temperature34 <= 150
 mul_constr35: 0.0116635708827124 t35 + mul_var35 = 0
 temperature_constr35: - 105.831091297194 exp_term35 + temperature35
   = 97.16890870280604
 temp_lower_35: temperature35 >= 90
 temp_upper_35: temperature35 <= 150
 mul_constr36: 0.0105051632041596 t36 + mul_var36 = 0
 temperature_constr36: - 112.0732669547247 exp_term36 + temperature36
   = 93.92673304527526
 temp_lower_36: temperature36 >= 90
 temp_upper_36: temperature36 <= 150
 mul_constr37: 0.00955513865780049 t37 + mul_var37 = 0
 temperature_constr37: - 115.9338918998374 exp_term37 + temperature37
   = 91.06610810016257
 temp_lower_37: temperature37 >= 90
 temp_upper_37: temperature37 <= 150
 mul_constr38: 0.0135515623747671 t38 + mul_var38 = 0
 temperature_constr38: - 101.4035973448839 exp_term38 + temperature38
   = 103.5964026551161
 temp_lower_38: temperature38 >= 90
 temp_upper_38: temperature38 <= 150
 mul_constr39: 0.00935712031425188 t39 + mul_var39 = 0
 temperature_constr39: - 124.2747780956032 exp_term39 + temperature39
   = 82.72522190439682
 temp_lower_39: temperature39 >= 90
 temp_upper_39: temperature39 <= 150
 mul_constr40: 0.00958054103605975 t40 + mul_var40 = 0
 temperature_constr40: - 117.6248546611039 exp_term40 + temperature40
   = 89.37514533889609
 temp_lower_40: temperature40 >= 90
 temp_upper_40: temperature40 <= 150
 mul_constr41: 0.0128669309492976 t41 + mul_var41 = 0
 temperature_constr41: - 103.5858355686346 exp_term41 + temperature41
   = 101.4141644313654
 temp_lower_41: temperature41 >= 90
 temp_upper_41: temperature41 <= 150
 mul_constr42: 0.0100209801065354 t42 + mul_var42 = 0
 temperature_constr42: - 115.9010419190696 exp_term42 + temperature42
   = 91.09895808093042
 temp_lower_42: temperature42 >= 90
 temp_upper_42: temperature42 <= 150
 mul_constr43: 0.0092335593933891 t43 + mul_var43 = 0
 temperature_constr43: - 123.4074412275519 exp_term43 + temperature43
   = 84.59255877244809
 temp_lower_43: temperature43 >= 90
 temp_upper_43: temperature43 <= 150
 mul_constr44: 0.0121229409569728 t44 + mul_var44 = 0
 temperature_constr44: - 107.1543031891656 exp_term44 + temperature44
   = 99.84569681083435
 temp_lower_44: temperature44 >= 90
 temp_upper_44: temperature44 <= 150
 mul_constr45: 0.0170896155716006 t45 + mul_var45 = 0
 temperature_constr45: - 87.44861450934243 exp_term45 + temperature45
   = 114.5513854906576
 temp_lower_45: temperature45 >= 90
 temp_upper_45: temperature45 <= 150
 mul_constr46: 0.0166693370230336 t46 + mul_var46 = 0
 temperature_constr46: - 91.15700758143295 exp_term46 + temperature46
   = 112.8429924185671
 temp_lower_46: temperature46 >= 90
 temp_upper_46: temperature46 <= 150
 mul_constr47: 0.0179760322113699 t47 + mul_var47 = 0
 temperature_constr47: - 90.06718170290705 exp_term47 + temperature47
   = 113.932818297093
 temp_lower_47: temperature47 >= 90
 temp_upper_47: temperature47 <= 150
 mul_constr48: 0.0171034378137564 t48 + mul_var48 = 0
 temperature_constr48: - 91.69803084683289 exp_term48 + temperature48
   = 113.3019691531671
 temp_lower_48: temperature48 >= 90
 temp_upper_48: temperature48 <= 150
 mul_constr49: 0.0186322174339244 t49 + mul_var49 = 0
 temperature_constr49: - 81.371956441943 exp_term49 + temperature49
   = 118.628043558057
 temp_lower_49: temperature49 >= 90
 temp_upper_49: temperature49 <= 150
 mul_constr50: 0.0167728341471456 t50 + mul_var50 = 0
 temperature_constr50: - 91.31712847210329 exp_term50 + temperature50
   = 113.6828715278967
 temp_lower_50: temperature50 >= 90
 temp_upper_50: temperature50 <= 150
 mul_constr51: 0.0161053049294895 t51 + mul_var51 = 0
 temperature_constr51: - 94.67132577072967 exp_term51 + temperature51
   = 109.3286742292703
 temp_lower_51: temperature51 >= 90
 temp_upper_51: temperature51 <= 150
 mul_constr52: 0.0153648814794203 t52 + mul_var52 = 0
 temperature_constr52: - 98.40436003188577 exp_term52 + temperature52
   = 107.5956399681142
 temp_lower_52: temperature52 >= 90
 temp_upper_52: temperature52 <= 150
 mul_constr53: 0.0168033884855486 t53 + mul_var53 = 0
 temperature_constr53: - 92.47595969901796 exp_term53 + temperature53
   = 112.524040300982
 temp_lower_53: temperature53 >= 90
 temp_upper_53: temperature53 <= 150
 inv_constr_0: [ v0 * y0 ] = 1
 inv_constr_1: [ v1 * y1 ] = 1
 inv_constr_2: [ v2 * y2 ] = 1
 inv_constr_3: [ v3 * y3 ] = 1
 inv_constr_4: [ v4 * y4 ] = 1
 inv_constr_5: [ v5 * y5 ] = 1
 inv_constr_6: [ v6 * y6 ] = 1
 inv_constr_7: [ v7 * y7 ] = 1
 inv_constr_8: [ v8 * y8 ] = 1
 inv_constr_9: [ v9 * y9 ] = 1
 inv_constr_10: [ v10 * y10 ] = 1
 inv_constr_11: [ v11 * y11 ] = 1
 inv_constr_12: [ v12 * y12 ] = 1
 inv_constr_13: [ v13 * y13 ] = 1
 inv_constr_14: [ v14 * y14 ] = 1
 inv_constr_15: [ v15 * y15 ] = 1
 inv_constr_16: [ v16 * y16 ] = 1
 inv_constr_17: [ v17 * y17 ] = 1
 inv_constr_18: [ v18 * y18 ] = 1
 inv_constr_19: [ v19 * y19 ] = 1
 inv_constr_20: [ v20 * y20 ] = 1
 inv_constr_21: [ v21 * y21 ] = 1
 inv_constr_22: [ v22 * y22 ] = 1
 inv_constr_23: [ v23 * y23 ] = 1
 inv_constr_24: [ v24 * y24 ] = 1
 inv_constr_25: [ v25 * y25 ] = 1
 inv_constr_26: [ v26 * y26 ] = 1
 inv_constr_27: [ v27 * y27 ] = 1
 inv_constr_28: [ v28 * y28 ] = 1
 inv_constr_29: [ v29 * y29 ] = 1
 inv_constr_30: [ v30 * y30 ] = 1
 inv_constr_31: [ v31 * y31 ] = 1
 inv_constr_32: [ v32 * y32 ] = 1
 inv_constr_33: [ v33 * y33 ] = 1
 inv_constr_34: [ v34 * y34 ] = 1
 inv_constr_35: [ v35 * y35 ] = 1
 inv_constr_36: [ v36 * y36 ] = 1
 inv_constr_37: [ v37 * y37 ] = 1
 inv_constr_38: [ v38 * y38 ] = 1
 inv_constr_39: [ v39 * y39 ] = 1
 inv_constr_40: [ v40 * y40 ] = 1
 inv_constr_41: [ v41 * y41 ] = 1
 inv_constr_42: [ v42 * y42 ] = 1
 inv_constr_43: [ v43 * y43 ] = 1
 inv_constr_44: [ v44 * y44 ] = 1
 inv_constr_45: [ v45 * y45 ] = 1
 inv_constr_46: [ v46 * y46 ] = 1
 inv_constr_47: [ v47 * y47 ] = 1
 inv_constr_48: [ v48 * y48 ] = 1
 inv_constr_49: [ v49 * y49 ] = 1
 inv_constr_50: [ v50 * y50 ] = 1
 inv_constr_51: [ v51 * y51 ] = 1
 inv_constr_52: [ v52 * y52 ] = 1
 inv_constr_53: [ v53 * y53 ] = 1
 v_z_relation_1: v1 + [ - 0.01 v0 * z1 ] = 0
 v_z_relation_2: v2 + [ - 0.01 v1 * z2 ] = 0
 v_z_relation_3: v3 + [ - 0.01 v2 * z3 ] = 0
 v_z_relation_4: v4 + [ - 0.01 v3 * z4 ] = 0
 v_z_relation_5: v5 + [ - 0.01 v4 * z5 ] = 0
 v_z_relation_6: v6 + [ - 0.01 v5 * z6 ] = 0
 v_z_relation_7: v7 + [ - 0.01 v6 * z7 ] = 0
 v_z_relation_8: v8 + [ - 0.01 v7 * z8 ] = 0
 v_z_relation_9: v9 + [ - 0.01 v8 * z9 ] = 0
 v_z_relation_10: v10 + [ - 0.01 v9 * z10 ] = 0
 v_z_relation_11: v11 + [ - 0.01 v10 * z11 ] = 0
 v_z_relation_12: v12 + [ - 0.01 v11 * z12 ] = 0
 v_z_relation_13: v13 + [ - 0.01 v12 * z13 ] = 0
 v_z_relation_14: v14 + [ - 0.01 v13 * z14 ] = 0
 v_z_relation_15: v15 + [ - 0.01 v14 * z15 ] = 0
 v_z_relation_16: v16 + [ - 0.01 v15 * z16 ] = 0
 v_z_relation_17: v17 + [ - 0.01 v16 * z17 ] = 0
 v_z_relation_18: v18 + [ - 0.01 v17 * z18 ] = 0
 v_z_relation_19: v19 + [ - 0.01 v18 * z19 ] = 0
 v_z_relation_20: v20 + [ - 0.01 v19 * z20 ] = 0
 v_z_relation_21: v21 + [ - 0.01 v20 * z21 ] = 0
 v_z_relation_22: v22 + [ - 0.01 v21 * z22 ] = 0
 v_z_relation_23: v23 + [ - 0.01 v22 * z23 ] = 0
 v_z_relation_24: v24 + [ - 0.01 v23 * z24 ] = 0
 v_z_relation_25: v25 + [ - 0.01 v24 * z25 ] = 0
 v_z_relation_26: v26 + [ - 0.01 v25 * z26 ] = 0
 v_z_relation_27: v27 + [ - 0.01 v26 * z27 ] = 0
 v_z_relation_28: v28 + [ - 0.01 v27 * z28 ] = 0
 v_z_relation_29: v29 + [ - 0.01 v28 * z29 ] = 0
 v_z_relation_30: v30 + [ - 0.01 v29 * z30 ] = 0
 v_z_relation_31: v31 + [ - 0.01 v30 * z31 ] = 0
 v_z_relation_32: v32 + [ - 0.01 v31 * z32 ] = 0
 v_z_relation_33: v33 + [ - 0.01 v32 * z33 ] = 0
 v_z_relation_34: v34 + [ - 0.01 v33 * z34 ] = 0
 v_z_relation_35: v35 + [ - 0.01 v34 * z35 ] = 0
 v_z_relation_36: v36 + [ - 0.01 v35 * z36 ] = 0
 v_z_relation_37: v37 + [ - 0.01 v36 * z37 ] = 0
 v_z_relation_38: v38 + [ - 0.01 v37 * z38 ] = 0
 v_z_relation_39: v39 + [ - 0.01 v38 * z39 ] = 0
 v_z_relation_40: v40 + [ - 0.01 v39 * z40 ] = 0
 v_z_relation_41: v41 + [ - 0.01 v40 * z41 ] = 0
 v_z_relation_42: v42 + [ - 0.01 v41 * z42 ] = 0
 v_z_relation_43: v43 + [ - 0.01 v42 * z43 ] = 0
 v_z_relation_44: v44 + [ - 0.01 v43 * z44 ] = 0
 v_z_relation_45: v45 + [ - 0.01 v44 * z45 ] = 0
 v_z_relation_46: v46 + [ - 0.01 v45 * z46 ] = 0
 v_z_relation_47: v47 + [ - 0.01 v46 * z47 ] = 0
 v_z_relation_48: v48 + [ - 0.01 v47 * z48 ] = 0
 v_z_relation_49: v49 + [ - 0.01 v48 * z49 ] = 0
 v_z_relation_50: v50 + [ - 0.01 v49 * z50 ] = 0
 v_z_relation_51: v51 + [ - 0.01 v50 * z51 ] = 0
 v_z_relation_52: v52 + [ - 0.01 v51 * z52 ] = 0
 v_z_relation_53: v53 + [ - 0.01 v52 * z53 ] = 0
Bounds
 5 <= v0 <= 100
 5 <= v1 <= 100
 5 <= v2 <= 100
 5 <= v3 <= 100
 5 <= v4 <= 100
 5 <= v5 <= 100
 5 <= v6 <= 100
 5 <= v7 <= 100
 5 <= v8 <= 100
 5 <= v9 <= 100
 5 <= v10 <= 100
 5 <= v11 <= 100
 5 <= v12 <= 100
 5 <= v13 <= 100
 5 <= v14 <= 100
 5 <= v15 <= 100
 5 <= v16 <= 100
 5 <= v17 <= 100
 5 <= v18 <= 100
 5 <= v19 <= 100
 5 <= v20 <= 100
 5 <= v21 <= 100
 5 <= v22 <= 100
 5 <= v23 <= 100
 5 <= v24 <= 100
 5 <= v25 <= 100
 5 <= v26 <= 100
 5 <= v27 <= 100
 5 <= v28 <= 100
 5 <= v29 <= 100
 5 <= v30 <= 100
 5 <= v31 <= 100
 5 <= v32 <= 100
 5 <= v33 <= 100
 5 <= v34 <= 100
 5 <= v35 <= 100
 5 <= v36 <= 100
 5 <= v37 <= 100
 5 <= v38 <= 100
 5 <= v39 <= 100
 5 <= v40 <= 100
 5 <= v41 <= 100
 5 <= v42 <= 100
 5 <= v43 <= 100
 5 <= v44 <= 100
 5 <= v45 <= 100
 5 <= v46 <= 100
 5 <= v47 <= 100
 5 <= v48 <= 100
 5 <= v49 <= 100
 5 <= v50 <= 100
 5 <= v51 <= 100
 5 <= v52 <= 100
 5 <= v53 <= 100
 t0 <= 500
 t1 <= 500
 t2 <= 500
 t3 <= 500
 t4 <= 500
 t5 <= 500
 t6 <= 500
 t7 <= 500
 t8 <= 500
 t9 <= 500
 t10 <= 500
 t11 <= 500
 t12 <= 500
 t13 <= 500
 t14 <= 500
 t15 <= 500
 t16 <= 500
 t17 <= 500
 t18 <= 500
 t19 <= 500
 t20 <= 500
 t21 <= 500
 t22 <= 500
 t23 <= 500
 t24 <= 500
 t25 <= 500
 t26 <= 500
 t27 <= 500
 t28 <= 500
 t29 <= 500
 t30 <= 500
 t31 <= 500
 t32 <= 500
 t33 <= 500
 t34 <= 500
 t35 <= 500
 t36 <= 500
 t37 <= 500
 t38 <= 500
 t39 <= 500
 t40 <= 500
 t41 <= 500
 t42 <= 500
 t43 <= 500
 t44 <= 500
 t45 <= 500
 t46 <= 500
 t47 <= 500
 t48 <= 500
 t49 <= 500
 t50 <= 500
 t51 <= 500
 t52 <= 500
 t53 <= 500
 0.01 <= y0 <= 0.2
 0.01 <= y1 <= 0.2
 0.01 <= y2 <= 0.2
 0.01 <= y3 <= 0.2
 0.01 <= y4 <= 0.2
 0.01 <= y5 <= 0.2
 0.01 <= y6 <= 0.2
 0.01 <= y7 <= 0.2
 0.01 <= y8 <= 0.2
 0.01 <= y9 <= 0.2
 0.01 <= y10 <= 0.2
 0.01 <= y11 <= 0.2
 0.01 <= y12 <= 0.2
 0.01 <= y13 <= 0.2
 0.01 <= y14 <= 0.2
 0.01 <= y15 <= 0.2
 0.01 <= y16 <= 0.2
 0.01 <= y17 <= 0.2
 0.01 <= y18 <= 0.2
 0.01 <= y19 <= 0.2
 0.01 <= y20 <= 0.2
 0.01 <= y21 <= 0.2
 0.01 <= y22 <= 0.2
 0.01 <= y23 <= 0.2
 0.01 <= y24 <= 0.2
 0.01 <= y25 <= 0.2
 0.01 <= y26 <= 0.2
 0.01 <= y27 <= 0.2
 0.01 <= y28 <= 0.2
 0.01 <= y29 <= 0.2
 0.01 <= y30 <= 0.2
 0.01 <= y31 <= 0.2
 0.01 <= y32 <= 0.2
 0.01 <= y33 <= 0.2
 0.01 <= y34 <= 0.2
 0.01 <= y35 <= 0.2
 0.01 <= y36 <= 0.2
 0.01 <= y37 <= 0.2
 0.01 <= y38 <= 0.2
 0.01 <= y39 <= 0.2
 0.01 <= y40 <= 0.2
 0.01 <= y41 <= 0.2
 0.01 <= y42 <= 0.2
 0.01 <= y43 <= 0.2
 0.01 <= y44 <= 0.2
 0.01 <= y45 <= 0.2
 0.01 <= y46 <= 0.2
 0.01 <= y47 <= 0.2
 0.01 <= y48 <= 0.2
 0.01 <= y49 <= 0.2
 0.01 <= y50 <= 0.2
 0.01 <= y51 <= 0.2
 0.01 <= y52 <= 0.2
 0.01 <= y53 <= 0.2
 95 <= z0 <= 105
 95 <= z1 <= 105
 95 <= z2 <= 105
 95 <= z3 <= 105
 95 <= z4 <= 105
 95 <= z5 <= 105
 95 <= z6 <= 105
 95 <= z7 <= 105
 95 <= z8 <= 105
 95 <= z9 <= 105
 95 <= z10 <= 105
 95 <= z11 <= 105
 95 <= z12 <= 105
 95 <= z13 <= 105
 95 <= z14 <= 105
 95 <= z15 <= 105
 95 <= z16 <= 105
 95 <= z17 <= 105
 95 <= z18 <= 105
 95 <= z19 <= 105
 95 <= z20 <= 105
 95 <= z21 <= 105
 95 <= z22 <= 105
 95 <= z23 <= 105
 95 <= z24 <= 105
 95 <= z25 <= 105
 95 <= z26 <= 105
 95 <= z27 <= 105
 95 <= z28 <= 105
 95 <= z29 <= 105
 95 <= z30 <= 105
 95 <= z31 <= 105
 95 <= z32 <= 105
 95 <= z33 <= 105
 95 <= z34 <= 105
 95 <= z35 <= 105
 95 <= z36 <= 105
 95 <= z37 <= 105
 95 <= z38 <= 105
 95 <= z39 <= 105
 95 <= z40 <= 105
 95 <= z41 <= 105
 95 <= z42 <= 105
 95 <= z43 <= 105
 95 <= z44 <= 105
 95 <= z45 <= 105
 95 <= z46 <= 105
 95 <= z47 <= 105
 95 <= z48 <= 105
 95 <= z49 <= 105
 95 <= z50 <= 105
 95 <= z51 <= 105
 95 <= z52 <= 105
 95 <= z53 <= 105
 -10 <= mul_var0 <= 0
 -10 <= mul_var1 <= 0
 -10 <= mul_var2 <= 0
 -10 <= mul_var3 <= 0
 -10 <= mul_var4 <= 0
 -10 <= mul_var5 <= 0
 -10 <= mul_var6 <= 0
 -10 <= mul_var7 <= 0
 -10 <= mul_var8 <= 0
 -10 <= mul_var9 <= 0
 -10 <= mul_var10 <= 0
 -10 <= mul_var11 <= 0
 -10 <= mul_var12 <= 0
 -10 <= mul_var13 <= 0
 -10 <= mul_var14 <= 0
 -10 <= mul_var15 <= 0
 -10 <= mul_var16 <= 0
 -10 <= mul_var17 <= 0
 -10 <= mul_var18 <= 0
 -10 <= mul_var19 <= 0
 -10 <= mul_var20 <= 0
 -10 <= mul_var21 <= 0
 -10 <= mul_var22 <= 0
 -10 <= mul_var23 <= 0
 -10 <= mul_var24 <= 0
 -10 <= mul_var25 <= 0
 -10 <= mul_var26 <= 0
 -10 <= mul_var27 <= 0
 -10 <= mul_var28 <= 0
 -10 <= mul_var29 <= 0
 -10 <= mul_var30 <= 0
 -10 <= mul_var31 <= 0
 -10 <= mul_var32 <= 0
 -10 <= mul_var33 <= 0
 -10 <= mul_var34 <= 0
 -10 <= mul_var35 <= 0
 -10 <= mul_var36 <= 0
 -10 <= mul_var37 <= 0
 -10 <= mul_var38 <= 0
 -10 <= mul_var39 <= 0
 -10 <= mul_var40 <= 0
 -10 <= mul_var41 <= 0
 -10 <= mul_var42 <= 0
 -10 <= mul_var43 <= 0
 -10 <= mul_var44 <= 0
 -10 <= mul_var45 <= 0
 -10 <= mul_var46 <= 0
 -10 <= mul_var47 <= 0
 -10 <= mul_var48 <= 0
 -10 <= mul_var49 <= 0
 -10 <= mul_var50 <= 0
 -10 <= mul_var51 <= 0
 -10 <= mul_var52 <= 0
 -10 <= mul_var53 <= 0
 Constant = 1
Generals
 z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15 z16 z17 z18 z19 z20
 z21 z22 z23 z24 z25 z26 z27 z28 z29 z30 z31 z32 z33 z34 z35 z36 z37 z38
 z39 z40 z41 z42 z43 z44 z45 z46 z47 z48 z49 z50 z51 z52 z53
General Constraints
 exp_constr0: exp_term0 = EXP ( mul_var0 )
 exp_constr1: exp_term1 = EXP ( mul_var1 )
 exp_constr2: exp_term2 = EXP ( mul_var2 )
 exp_constr3: exp_term3 = EXP ( mul_var3 )
 exp_constr4: exp_term4 = EXP ( mul_var4 )
 exp_constr5: exp_term5 = EXP ( mul_var5 )
 exp_constr6: exp_term6 = EXP ( mul_var6 )
 exp_constr7: exp_term7 = EXP ( mul_var7 )
 exp_constr8: exp_term8 = EXP ( mul_var8 )
 exp_constr9: exp_term9 = EXP ( mul_var9 )
 exp_constr10: exp_term10 = EXP ( mul_var10 )
 exp_constr11: exp_term11 = EXP ( mul_var11 )
 exp_constr12: exp_term12 = EXP ( mul_var12 )
 exp_constr13: exp_term13 = EXP ( mul_var13 )
 exp_constr14: exp_term14 = EXP ( mul_var14 )
 exp_constr15: exp_term15 = EXP ( mul_var15 )
 exp_constr16: exp_term16 = EXP ( mul_var16 )
 exp_constr17: exp_term17 = EXP ( mul_var17 )
 exp_constr18: exp_term18 = EXP ( mul_var18 )
 exp_constr19: exp_term19 = EXP ( mul_var19 )
 exp_constr20: exp_term20 = EXP ( mul_var20 )
 exp_constr21: exp_term21 = EXP ( mul_var21 )
 exp_constr22: exp_term22 = EXP ( mul_var22 )
 exp_constr23: exp_term23 = EXP ( mul_var23 )
 exp_constr24: exp_term24 = EXP ( mul_var24 )
 exp_constr25: exp_term25 = EXP ( mul_var25 )
 exp_constr26: exp_term26 = EXP ( mul_var26 )
 exp_constr27: exp_term27 = EXP ( mul_var27 )
 exp_constr28: exp_term28 = EXP ( mul_var28 )
 exp_constr29: exp_term29 = EXP ( mul_var29 )
 exp_constr30: exp_term30 = EXP ( mul_var30 )
 exp_constr31: exp_term31 = EXP ( mul_var31 )
 exp_constr32: exp_term32 = EXP ( mul_var32 )
 exp_constr33: exp_term33 = EXP ( mul_var33 )
 exp_constr34: exp_term34 = EXP ( mul_var34 )
 exp_constr35: exp_term35 = EXP ( mul_var35 )
 exp_constr36: exp_term36 = EXP ( mul_var36 )
 exp_constr37: exp_term37 = EXP ( mul_var37 )
 exp_constr38: exp_term38 = EXP ( mul_var38 )
 exp_constr39: exp_term39 = EXP ( mul_var39 )
 exp_constr40: exp_term40 = EXP ( mul_var40 )
 exp_constr41: exp_term41 = EXP ( mul_var41 )
 exp_constr42: exp_term42 = EXP ( mul_var42 )
 exp_constr43: exp_term43 = EXP ( mul_var43 )
 exp_constr44: exp_term44 = EXP ( mul_var44 )
 exp_constr45: exp_term45 = EXP ( mul_var45 )
 exp_constr46: exp_term46 = EXP ( mul_var46 )
 exp_constr47: exp_term47 = EXP ( mul_var47 )
 exp_constr48: exp_term48 = EXP ( mul_var48 )
 exp_constr49: exp_term49 = EXP ( mul_var49 )
 exp_constr50: exp_term50 = EXP ( mul_var50 )
 exp_constr51: exp_term51 = EXP ( mul_var51 )
 exp_constr52: exp_term52 = EXP ( mul_var52 )
 exp_constr53: exp_term53 = EXP ( mul_var53 )
End
