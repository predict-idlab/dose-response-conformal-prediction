    function targMap = targDataMap(),

    ;%***********************
    ;% Create Parameter Map *
    ;%***********************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 2;
        sectIdxOffset = 0;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc paramMap
        ;%
        paramMap.nSections           = nTotSects;
        paramMap.sectIdxOffset       = sectIdxOffset;
            paramMap.sections(nTotSects) = dumSection; %prealloc
        paramMap.nTotData            = -1;

        ;%
        ;% Auto data (rtP)
        ;%
            section.nData     = 129;
            section.data(129)  = dumData; %prealloc

                    ;% rtP.Anest_loop
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

                    ;% rtP.Cobasis
                    section.data(2).logicalSrcIdx = 1;
                    section.data(2).dtTransOffset = 1;

                    ;% rtP.Dist_type
                    section.data(3).logicalSrcIdx = 2;
                    section.data(3).dtTransOffset = 2;

                    ;% rtP.MAPbasis
                    section.data(4).logicalSrcIdx = 3;
                    section.data(4).dtTransOffset = 3;

                    ;% rtP.RSM_type
                    section.data(5).logicalSrcIdx = 4;
                    section.data(5).dtTransOffset = 4;

                    ;% rtP.alpha_G
                    section.data(6).logicalSrcIdx = 5;
                    section.data(6).dtTransOffset = 5;

                    ;% rtP.alpha_RG
                    section.data(7).logicalSrcIdx = 6;
                    section.data(7).dtTransOffset = 6;

                    ;% rtP.Internal_A_pr
                    section.data(8).logicalSrcIdx = 7;
                    section.data(8).dtTransOffset = 7;

                    ;% rtP.Internal_B_pr
                    section.data(9).logicalSrcIdx = 8;
                    section.data(9).dtTransOffset = 8;

                    ;% rtP.Internal_C_pr
                    section.data(10).logicalSrcIdx = 9;
                    section.data(10).dtTransOffset = 9;

                    ;% rtP.Internal_InitialCondition
                    section.data(11).logicalSrcIdx = 10;
                    section.data(11).dtTransOffset = 10;

                    ;% rtP.TransferFcn_A
                    section.data(12).logicalSrcIdx = 11;
                    section.data(12).dtTransOffset = 11;

                    ;% rtP.TransferFcn_C
                    section.data(13).logicalSrcIdx = 12;
                    section.data(13).dtTransOffset = 12;

                    ;% rtP.Internal_A_pr_bxw4oadisy
                    section.data(14).logicalSrcIdx = 13;
                    section.data(14).dtTransOffset = 13;

                    ;% rtP.Internal_B_pr_acd2u0mx1v
                    section.data(15).logicalSrcIdx = 14;
                    section.data(15).dtTransOffset = 14;

                    ;% rtP.Internal_C_pr_cbzmhk4ru4
                    section.data(16).logicalSrcIdx = 15;
                    section.data(16).dtTransOffset = 15;

                    ;% rtP.Internal_InitialCondition_flhq4j1c44
                    section.data(17).logicalSrcIdx = 16;
                    section.data(17).dtTransOffset = 16;

                    ;% rtP.Saturation_UpperSat
                    section.data(18).logicalSrcIdx = 17;
                    section.data(18).dtTransOffset = 17;

                    ;% rtP.Saturation_LowerSat
                    section.data(19).logicalSrcIdx = 18;
                    section.data(19).dtTransOffset = 18;

                    ;% rtP.Internal_A_pr_oxglitxgdi
                    section.data(20).logicalSrcIdx = 19;
                    section.data(20).dtTransOffset = 19;

                    ;% rtP.Internal_B_pr_ixcvq1t1i0
                    section.data(21).logicalSrcIdx = 20;
                    section.data(21).dtTransOffset = 20;

                    ;% rtP.Internal_C_pr_a30ee2albx
                    section.data(22).logicalSrcIdx = 21;
                    section.data(22).dtTransOffset = 21;

                    ;% rtP.Internal_InitialCondition_plhlb04yjz
                    section.data(23).logicalSrcIdx = 22;
                    section.data(23).dtTransOffset = 22;

                    ;% rtP.Saturation_UpperSat_kjur3awdxo
                    section.data(24).logicalSrcIdx = 23;
                    section.data(24).dtTransOffset = 23;

                    ;% rtP.Saturation_LowerSat_i3sbtd4ihe
                    section.data(25).logicalSrcIdx = 24;
                    section.data(25).dtTransOffset = 24;

                    ;% rtP.Saturation_UpperSat_ce4kz1xeh2
                    section.data(26).logicalSrcIdx = 25;
                    section.data(26).dtTransOffset = 25;

                    ;% rtP.Saturation_LowerSat_j0oq522fxt
                    section.data(27).logicalSrcIdx = 26;
                    section.data(27).dtTransOffset = 26;

                    ;% rtP.TransferFcn_A_c31gaddzxu
                    section.data(28).logicalSrcIdx = 27;
                    section.data(28).dtTransOffset = 27;

                    ;% rtP.TransferFcn_C_pwm5fjxov3
                    section.data(29).logicalSrcIdx = 28;
                    section.data(29).dtTransOffset = 28;

                    ;% rtP.TransferFcn_A_cgq1qqeziw
                    section.data(30).logicalSrcIdx = 29;
                    section.data(30).dtTransOffset = 29;

                    ;% rtP.TransferFcn_C_kg3euht54j
                    section.data(31).logicalSrcIdx = 30;
                    section.data(31).dtTransOffset = 30;

                    ;% rtP.Internal_A_pr_nkheacrqmt
                    section.data(32).logicalSrcIdx = 31;
                    section.data(32).dtTransOffset = 31;

                    ;% rtP.Internal_B_pr_l20xfqn4io
                    section.data(33).logicalSrcIdx = 32;
                    section.data(33).dtTransOffset = 40;

                    ;% rtP.Internal_C_pr_ihduuzxooi
                    section.data(34).logicalSrcIdx = 33;
                    section.data(34).dtTransOffset = 41;

                    ;% rtP.Internal_InitialCondition_f013nqgro1
                    section.data(35).logicalSrcIdx = 34;
                    section.data(35).dtTransOffset = 42;

                    ;% rtP.Gain1_Gain
                    section.data(36).logicalSrcIdx = 35;
                    section.data(36).dtTransOffset = 43;

                    ;% rtP.Internal_A_pr_n4kfo5ljxc
                    section.data(37).logicalSrcIdx = 36;
                    section.data(37).dtTransOffset = 44;

                    ;% rtP.Internal_B_pr_l0u03c0wog
                    section.data(38).logicalSrcIdx = 37;
                    section.data(38).dtTransOffset = 53;

                    ;% rtP.Internal_C_pr_coj5mfz3fg
                    section.data(39).logicalSrcIdx = 38;
                    section.data(39).dtTransOffset = 54;

                    ;% rtP.Internal_InitialCondition_dascwzc0ku
                    section.data(40).logicalSrcIdx = 39;
                    section.data(40).dtTransOffset = 55;

                    ;% rtP.Gain2_Gain
                    section.data(41).logicalSrcIdx = 40;
                    section.data(41).dtTransOffset = 56;

                    ;% rtP.Saturation7_UpperSat
                    section.data(42).logicalSrcIdx = 41;
                    section.data(42).dtTransOffset = 57;

                    ;% rtP.Saturation7_LowerSat
                    section.data(43).logicalSrcIdx = 42;
                    section.data(43).dtTransOffset = 58;

                    ;% rtP.TransferFcn_A_eiu4to52sp
                    section.data(44).logicalSrcIdx = 43;
                    section.data(44).dtTransOffset = 59;

                    ;% rtP.TransferFcn_C_ccp0mwtknu
                    section.data(45).logicalSrcIdx = 44;
                    section.data(45).dtTransOffset = 60;

                    ;% rtP.TransferFcn_A_fdwj0zhm3r
                    section.data(46).logicalSrcIdx = 45;
                    section.data(46).dtTransOffset = 61;

                    ;% rtP.TransferFcn_C_pcze5nejeb
                    section.data(47).logicalSrcIdx = 46;
                    section.data(47).dtTransOffset = 62;

                    ;% rtP.Internal_A_pr_gtcc2nztln
                    section.data(48).logicalSrcIdx = 47;
                    section.data(48).dtTransOffset = 63;

                    ;% rtP.Internal_B_pr_e0vgcru5yv
                    section.data(49).logicalSrcIdx = 48;
                    section.data(49).dtTransOffset = 72;

                    ;% rtP.Internal_C_pr_nmldud5anh
                    section.data(50).logicalSrcIdx = 49;
                    section.data(50).dtTransOffset = 73;

                    ;% rtP.Internal_InitialCondition_fyztitczqh
                    section.data(51).logicalSrcIdx = 50;
                    section.data(51).dtTransOffset = 74;

                    ;% rtP.Internal_A_pr_ojosqlrsev
                    section.data(52).logicalSrcIdx = 51;
                    section.data(52).dtTransOffset = 75;

                    ;% rtP.Internal_B_pr_isexhvxkqn
                    section.data(53).logicalSrcIdx = 52;
                    section.data(53).dtTransOffset = 84;

                    ;% rtP.Internal_C_pr_hhrpjg032d
                    section.data(54).logicalSrcIdx = 53;
                    section.data(54).dtTransOffset = 85;

                    ;% rtP.Internal_InitialCondition_eoiqws2jvx
                    section.data(55).logicalSrcIdx = 54;
                    section.data(55).dtTransOffset = 86;

                    ;% rtP.Saturation8_UpperSat
                    section.data(56).logicalSrcIdx = 55;
                    section.data(56).dtTransOffset = 87;

                    ;% rtP.Saturation8_LowerSat
                    section.data(57).logicalSrcIdx = 56;
                    section.data(57).dtTransOffset = 88;

                    ;% rtP.Internal_A_pr_i2ltighkdc
                    section.data(58).logicalSrcIdx = 57;
                    section.data(58).dtTransOffset = 89;

                    ;% rtP.Internal_B_pr_mlvsl1pdjw
                    section.data(59).logicalSrcIdx = 58;
                    section.data(59).dtTransOffset = 94;

                    ;% rtP.Internal_C_pr_puxizhhqsn
                    section.data(60).logicalSrcIdx = 59;
                    section.data(60).dtTransOffset = 95;

                    ;% rtP.Internal_InitialCondition_nizog25s2h
                    section.data(61).logicalSrcIdx = 60;
                    section.data(61).dtTransOffset = 96;

                    ;% rtP.Saturation9_UpperSat
                    section.data(62).logicalSrcIdx = 61;
                    section.data(62).dtTransOffset = 97;

                    ;% rtP.Saturation9_LowerSat
                    section.data(63).logicalSrcIdx = 62;
                    section.data(63).dtTransOffset = 98;

                    ;% rtP.Internal_A_pr_m4zhmj0l23
                    section.data(64).logicalSrcIdx = 63;
                    section.data(64).dtTransOffset = 99;

                    ;% rtP.Internal_B_pr_aopwchlqyq
                    section.data(65).logicalSrcIdx = 64;
                    section.data(65).dtTransOffset = 100;

                    ;% rtP.Internal_C_pr_puewk12or5
                    section.data(66).logicalSrcIdx = 65;
                    section.data(66).dtTransOffset = 101;

                    ;% rtP.Internal_InitialCondition_oekgjvagkb
                    section.data(67).logicalSrcIdx = 66;
                    section.data(67).dtTransOffset = 102;

                    ;% rtP.Saturation6_UpperSat
                    section.data(68).logicalSrcIdx = 67;
                    section.data(68).dtTransOffset = 103;

                    ;% rtP.Saturation6_LowerSat
                    section.data(69).logicalSrcIdx = 68;
                    section.data(69).dtTransOffset = 104;

                    ;% rtP.Atracuriuminput_Time0
                    section.data(70).logicalSrcIdx = 69;
                    section.data(70).dtTransOffset = 105;

                    ;% rtP.Atracuriuminput_Data0
                    section.data(71).logicalSrcIdx = 70;
                    section.data(71).dtTransOffset = 415;

                    ;% rtP.Dopamineinput_Time0
                    section.data(72).logicalSrcIdx = 71;
                    section.data(72).dtTransOffset = 725;

                    ;% rtP.Dopamineinput_Data0
                    section.data(73).logicalSrcIdx = 72;
                    section.data(73).dtTransOffset = 1035;

                    ;% rtP.Propofolinput_Time0
                    section.data(74).logicalSrcIdx = 73;
                    section.data(74).dtTransOffset = 1345;

                    ;% rtP.Propofolinput_Data0
                    section.data(75).logicalSrcIdx = 74;
                    section.data(75).dtTransOffset = 1655;

                    ;% rtP.Remifentanilinput_Time0
                    section.data(76).logicalSrcIdx = 75;
                    section.data(76).dtTransOffset = 1965;

                    ;% rtP.Remifentanilinput_Data0
                    section.data(77).logicalSrcIdx = 76;
                    section.data(77).dtTransOffset = 2275;

                    ;% rtP.SNPinput_Time0
                    section.data(78).logicalSrcIdx = 77;
                    section.data(78).dtTransOffset = 2585;

                    ;% rtP.SNPinput_Data0
                    section.data(79).logicalSrcIdx = 78;
                    section.data(79).dtTransOffset = 2895;

                    ;% rtP.Saturation2_UpperSat
                    section.data(80).logicalSrcIdx = 79;
                    section.data(80).dtTransOffset = 3205;

                    ;% rtP.Saturation2_LowerSat
                    section.data(81).logicalSrcIdx = 80;
                    section.data(81).dtTransOffset = 3206;

                    ;% rtP._Gain
                    section.data(82).logicalSrcIdx = 81;
                    section.data(82).dtTransOffset = 3207;

                    ;% rtP.FromWorkspace1_Time0
                    section.data(83).logicalSrcIdx = 82;
                    section.data(83).dtTransOffset = 3208;

                    ;% rtP.FromWorkspace1_Data0
                    section.data(84).logicalSrcIdx = 83;
                    section.data(84).dtTransOffset = 4259;

                    ;% rtP.FromWorkspace_Time0
                    section.data(85).logicalSrcIdx = 84;
                    section.data(85).dtTransOffset = 5310;

                    ;% rtP.FromWorkspace_Data0
                    section.data(86).logicalSrcIdx = 85;
                    section.data(86).dtTransOffset = 6361;

                    ;% rtP.Circumcision_Time0
                    section.data(87).logicalSrcIdx = 86;
                    section.data(87).dtTransOffset = 7412;

                    ;% rtP.Circumcision_Data0
                    section.data(88).logicalSrcIdx = 87;
                    section.data(88).dtTransOffset = 7730;

                    ;% rtP.Nodisturbance_Time0
                    section.data(89).logicalSrcIdx = 88;
                    section.data(89).dtTransOffset = 8048;

                    ;% rtP.Nodisturbance_Data0
                    section.data(90).logicalSrcIdx = 89;
                    section.data(90).dtTransOffset = 8548;

                    ;% rtP.KneeArthroscopy_Time0
                    section.data(91).logicalSrcIdx = 90;
                    section.data(91).dtTransOffset = 9048;

                    ;% rtP.KneeArthroscopy_Data0
                    section.data(92).logicalSrcIdx = 91;
                    section.data(92).dtTransOffset = 9627;

                    ;% rtP.EpididymalCystectomy_Time0
                    section.data(93).logicalSrcIdx = 92;
                    section.data(93).dtTransOffset = 10206;

                    ;% rtP.EpididymalCystectomy_Data0
                    section.data(94).logicalSrcIdx = 93;
                    section.data(94).dtTransOffset = 10674;

                    ;% rtP.ScrotalExploration_Time0
                    section.data(95).logicalSrcIdx = 94;
                    section.data(95).dtTransOffset = 11142;

                    ;% rtP.ScrotalExploration_Data0
                    section.data(96).logicalSrcIdx = 95;
                    section.data(96).dtTransOffset = 11898;

                    ;% rtP.Myomectomy_Time0
                    section.data(97).logicalSrcIdx = 96;
                    section.data(97).dtTransOffset = 12654;

                    ;% rtP.Myomectomy_Data0
                    section.data(98).logicalSrcIdx = 97;
                    section.data(98).dtTransOffset = 14190;

                    ;% rtP.Urethroplasty_Time0
                    section.data(99).logicalSrcIdx = 98;
                    section.data(99).dtTransOffset = 15726;

                    ;% rtP.Urethroplasty_Data0
                    section.data(100).logicalSrcIdx = 99;
                    section.data(100).dtTransOffset = 17550;

                    ;% rtP.Mammoplasty_Time0
                    section.data(101).logicalSrcIdx = 100;
                    section.data(101).dtTransOffset = 19374;

                    ;% rtP.Mammoplasty_Data0
                    section.data(102).logicalSrcIdx = 101;
                    section.data(102).dtTransOffset = 20718;

                    ;% rtP.OpenCholecysectomy_Time0
                    section.data(103).logicalSrcIdx = 102;
                    section.data(103).dtTransOffset = 22062;

                    ;% rtP.OpenCholecysectomy_Data0
                    section.data(104).logicalSrcIdx = 103;
                    section.data(104).dtTransOffset = 23286;

                    ;% rtP.Vasovasostomy_Time0
                    section.data(105).logicalSrcIdx = 104;
                    section.data(105).dtTransOffset = 24510;

                    ;% rtP.Vasovasostomy_Data0
                    section.data(106).logicalSrcIdx = 105;
                    section.data(106).dtTransOffset = 26670;

                    ;% rtP.Hysterectomy_Time0
                    section.data(107).logicalSrcIdx = 106;
                    section.data(107).dtTransOffset = 28830;

                    ;% rtP.Hysterectomy_Data0
                    section.data(108).logicalSrcIdx = 107;
                    section.data(108).dtTransOffset = 30258;

                    ;% rtP.Internal_A_pr_kx5sallgve
                    section.data(109).logicalSrcIdx = 108;
                    section.data(109).dtTransOffset = 31686;

                    ;% rtP.Internal_B_pr_hfxj0kvrnc
                    section.data(110).logicalSrcIdx = 109;
                    section.data(110).dtTransOffset = 31693;

                    ;% rtP.Internal_C_pr_av2jlvpum0
                    section.data(111).logicalSrcIdx = 110;
                    section.data(111).dtTransOffset = 31694;

                    ;% rtP.Internal_InitialCondition_lpky2t20lr
                    section.data(112).logicalSrcIdx = 111;
                    section.data(112).dtTransOffset = 31695;

                    ;% rtP.Internal_A_pr_m25t4b112u
                    section.data(113).logicalSrcIdx = 112;
                    section.data(113).dtTransOffset = 31696;

                    ;% rtP.Internal_B_pr_kda05yf4jy
                    section.data(114).logicalSrcIdx = 113;
                    section.data(114).dtTransOffset = 31703;

                    ;% rtP.Internal_C_pr_klda0set0f
                    section.data(115).logicalSrcIdx = 114;
                    section.data(115).dtTransOffset = 31704;

                    ;% rtP.Internal_InitialCondition_lwwah2kv3v
                    section.data(116).logicalSrcIdx = 115;
                    section.data(116).dtTransOffset = 31705;

                    ;% rtP.Internal_A_pr_puchlpq5ne
                    section.data(117).logicalSrcIdx = 116;
                    section.data(117).dtTransOffset = 31706;

                    ;% rtP.Internal_B_pr_d3jkwmc1kb
                    section.data(118).logicalSrcIdx = 117;
                    section.data(118).dtTransOffset = 31707;

                    ;% rtP.Internal_C_pr_cnq1if3xbe
                    section.data(119).logicalSrcIdx = 118;
                    section.data(119).dtTransOffset = 31708;

                    ;% rtP.Internal_InitialCondition_cbt5hmxici
                    section.data(120).logicalSrcIdx = 119;
                    section.data(120).dtTransOffset = 31709;

                    ;% rtP.Saturation1_UpperSat
                    section.data(121).logicalSrcIdx = 120;
                    section.data(121).dtTransOffset = 31710;

                    ;% rtP.Saturation1_LowerSat
                    section.data(122).logicalSrcIdx = 121;
                    section.data(122).dtTransOffset = 31711;

                    ;% rtP.Saturation3_UpperSat
                    section.data(123).logicalSrcIdx = 122;
                    section.data(123).dtTransOffset = 31712;

                    ;% rtP.Saturation3_LowerSat
                    section.data(124).logicalSrcIdx = 123;
                    section.data(124).dtTransOffset = 31713;

                    ;% rtP.Saturation4_UpperSat
                    section.data(125).logicalSrcIdx = 124;
                    section.data(125).dtTransOffset = 31714;

                    ;% rtP.Saturation4_LowerSat
                    section.data(126).logicalSrcIdx = 125;
                    section.data(126).dtTransOffset = 31715;

                    ;% rtP.patientweight_Gain
                    section.data(127).logicalSrcIdx = 126;
                    section.data(127).dtTransOffset = 31716;

                    ;% rtP.Saturation5_UpperSat
                    section.data(128).logicalSrcIdx = 127;
                    section.data(128).dtTransOffset = 31717;

                    ;% rtP.Saturation5_LowerSat
                    section.data(129).logicalSrcIdx = 128;
                    section.data(129).dtTransOffset = 31718;

            nTotData = nTotData + section.nData;
            paramMap.sections(1) = section;
            clear section

            section.nData     = 72;
            section.data(72)  = dumData; %prealloc

                    ;% rtP.Internal_A_ir
                    section.data(1).logicalSrcIdx = 129;
                    section.data(1).dtTransOffset = 0;

                    ;% rtP.Internal_A_jc
                    section.data(2).logicalSrcIdx = 130;
                    section.data(2).dtTransOffset = 1;

                    ;% rtP.Internal_B_ir
                    section.data(3).logicalSrcIdx = 131;
                    section.data(3).dtTransOffset = 3;

                    ;% rtP.Internal_B_jc
                    section.data(4).logicalSrcIdx = 132;
                    section.data(4).dtTransOffset = 4;

                    ;% rtP.Internal_C_ir
                    section.data(5).logicalSrcIdx = 133;
                    section.data(5).dtTransOffset = 6;

                    ;% rtP.Internal_C_jc
                    section.data(6).logicalSrcIdx = 134;
                    section.data(6).dtTransOffset = 7;

                    ;% rtP.Internal_A_ir_hf5tox2xk5
                    section.data(7).logicalSrcIdx = 135;
                    section.data(7).dtTransOffset = 9;

                    ;% rtP.Internal_A_jc_ogadgmqmoo
                    section.data(8).logicalSrcIdx = 136;
                    section.data(8).dtTransOffset = 10;

                    ;% rtP.Internal_B_ir_oqjkicywhm
                    section.data(9).logicalSrcIdx = 137;
                    section.data(9).dtTransOffset = 12;

                    ;% rtP.Internal_B_jc_bbjjxvfjvn
                    section.data(10).logicalSrcIdx = 138;
                    section.data(10).dtTransOffset = 13;

                    ;% rtP.Internal_C_ir_akgglybrpk
                    section.data(11).logicalSrcIdx = 139;
                    section.data(11).dtTransOffset = 15;

                    ;% rtP.Internal_C_jc_jfpd0pnvih
                    section.data(12).logicalSrcIdx = 140;
                    section.data(12).dtTransOffset = 16;

                    ;% rtP.Internal_A_ir_dgoldd154g
                    section.data(13).logicalSrcIdx = 141;
                    section.data(13).dtTransOffset = 18;

                    ;% rtP.Internal_A_jc_pge3o4iqct
                    section.data(14).logicalSrcIdx = 142;
                    section.data(14).dtTransOffset = 19;

                    ;% rtP.Internal_B_ir_k0filihusw
                    section.data(15).logicalSrcIdx = 143;
                    section.data(15).dtTransOffset = 21;

                    ;% rtP.Internal_B_jc_onaxgo32ml
                    section.data(16).logicalSrcIdx = 144;
                    section.data(16).dtTransOffset = 22;

                    ;% rtP.Internal_C_ir_iyqg3glnsf
                    section.data(17).logicalSrcIdx = 145;
                    section.data(17).dtTransOffset = 24;

                    ;% rtP.Internal_C_jc_llex4eusi2
                    section.data(18).logicalSrcIdx = 146;
                    section.data(18).dtTransOffset = 25;

                    ;% rtP.Internal_A_ir_kh1ahh4kw5
                    section.data(19).logicalSrcIdx = 147;
                    section.data(19).dtTransOffset = 27;

                    ;% rtP.Internal_A_jc_cd0lveyter
                    section.data(20).logicalSrcIdx = 148;
                    section.data(20).dtTransOffset = 36;

                    ;% rtP.Internal_B_ir_nrfq1m3t4y
                    section.data(21).logicalSrcIdx = 149;
                    section.data(21).dtTransOffset = 42;

                    ;% rtP.Internal_B_jc_ac1tk1l1cm
                    section.data(22).logicalSrcIdx = 150;
                    section.data(22).dtTransOffset = 43;

                    ;% rtP.Internal_C_ir_o11oddr1av
                    section.data(23).logicalSrcIdx = 151;
                    section.data(23).dtTransOffset = 45;

                    ;% rtP.Internal_C_jc_d2vn1pgejl
                    section.data(24).logicalSrcIdx = 152;
                    section.data(24).dtTransOffset = 46;

                    ;% rtP.Internal_A_ir_lq4j4xfz2n
                    section.data(25).logicalSrcIdx = 153;
                    section.data(25).dtTransOffset = 52;

                    ;% rtP.Internal_A_jc_acfygnacds
                    section.data(26).logicalSrcIdx = 154;
                    section.data(26).dtTransOffset = 61;

                    ;% rtP.Internal_B_ir_ijyvpjmjsq
                    section.data(27).logicalSrcIdx = 155;
                    section.data(27).dtTransOffset = 67;

                    ;% rtP.Internal_B_jc_aajyzu2cah
                    section.data(28).logicalSrcIdx = 156;
                    section.data(28).dtTransOffset = 68;

                    ;% rtP.Internal_C_ir_ky5xf0i1px
                    section.data(29).logicalSrcIdx = 157;
                    section.data(29).dtTransOffset = 70;

                    ;% rtP.Internal_C_jc_galqf04ynt
                    section.data(30).logicalSrcIdx = 158;
                    section.data(30).dtTransOffset = 71;

                    ;% rtP.Internal_A_ir_aak1secaho
                    section.data(31).logicalSrcIdx = 159;
                    section.data(31).dtTransOffset = 77;

                    ;% rtP.Internal_A_jc_kanybgxs0e
                    section.data(32).logicalSrcIdx = 160;
                    section.data(32).dtTransOffset = 86;

                    ;% rtP.Internal_B_ir_d0aaprxg0k
                    section.data(33).logicalSrcIdx = 161;
                    section.data(33).dtTransOffset = 92;

                    ;% rtP.Internal_B_jc_gbxmowzmqz
                    section.data(34).logicalSrcIdx = 162;
                    section.data(34).dtTransOffset = 93;

                    ;% rtP.Internal_C_ir_ibqb01xp5c
                    section.data(35).logicalSrcIdx = 163;
                    section.data(35).dtTransOffset = 95;

                    ;% rtP.Internal_C_jc_dwqmfgpi2b
                    section.data(36).logicalSrcIdx = 164;
                    section.data(36).dtTransOffset = 96;

                    ;% rtP.Internal_A_ir_dpjnaf3xwz
                    section.data(37).logicalSrcIdx = 165;
                    section.data(37).dtTransOffset = 102;

                    ;% rtP.Internal_A_jc_n321zog0ex
                    section.data(38).logicalSrcIdx = 166;
                    section.data(38).dtTransOffset = 111;

                    ;% rtP.Internal_B_ir_kkrwmhr25e
                    section.data(39).logicalSrcIdx = 167;
                    section.data(39).dtTransOffset = 117;

                    ;% rtP.Internal_B_jc_bmh4dtrh0s
                    section.data(40).logicalSrcIdx = 168;
                    section.data(40).dtTransOffset = 118;

                    ;% rtP.Internal_C_ir_mjslb5eyrb
                    section.data(41).logicalSrcIdx = 169;
                    section.data(41).dtTransOffset = 120;

                    ;% rtP.Internal_C_jc_au0h0wd2zz
                    section.data(42).logicalSrcIdx = 170;
                    section.data(42).dtTransOffset = 121;

                    ;% rtP.Internal_A_ir_deilu3523u
                    section.data(43).logicalSrcIdx = 171;
                    section.data(43).dtTransOffset = 127;

                    ;% rtP.Internal_A_jc_nnhrvh4uhs
                    section.data(44).logicalSrcIdx = 172;
                    section.data(44).dtTransOffset = 132;

                    ;% rtP.Internal_B_ir_bf5h52djfe
                    section.data(45).logicalSrcIdx = 173;
                    section.data(45).dtTransOffset = 136;

                    ;% rtP.Internal_B_jc_g51vo4a3vx
                    section.data(46).logicalSrcIdx = 174;
                    section.data(46).dtTransOffset = 137;

                    ;% rtP.Internal_C_ir_j0bboxekne
                    section.data(47).logicalSrcIdx = 175;
                    section.data(47).dtTransOffset = 139;

                    ;% rtP.Internal_C_jc_ap3qqcgpmi
                    section.data(48).logicalSrcIdx = 176;
                    section.data(48).dtTransOffset = 140;

                    ;% rtP.Internal_A_ir_myylvo2i45
                    section.data(49).logicalSrcIdx = 177;
                    section.data(49).dtTransOffset = 144;

                    ;% rtP.Internal_A_jc_ig2cutmpb4
                    section.data(50).logicalSrcIdx = 178;
                    section.data(50).dtTransOffset = 145;

                    ;% rtP.Internal_B_ir_f5rfza45d4
                    section.data(51).logicalSrcIdx = 179;
                    section.data(51).dtTransOffset = 147;

                    ;% rtP.Internal_B_jc_lblmejayek
                    section.data(52).logicalSrcIdx = 180;
                    section.data(52).dtTransOffset = 148;

                    ;% rtP.Internal_C_ir_os2whdiotd
                    section.data(53).logicalSrcIdx = 181;
                    section.data(53).dtTransOffset = 150;

                    ;% rtP.Internal_C_jc_pwb3pumdzw
                    section.data(54).logicalSrcIdx = 182;
                    section.data(54).dtTransOffset = 151;

                    ;% rtP.Internal_A_ir_ijulsdwdmi
                    section.data(55).logicalSrcIdx = 183;
                    section.data(55).dtTransOffset = 153;

                    ;% rtP.Internal_A_jc_p4ci0hu5qn
                    section.data(56).logicalSrcIdx = 184;
                    section.data(56).dtTransOffset = 160;

                    ;% rtP.Internal_B_ir_jusrbow3ct
                    section.data(57).logicalSrcIdx = 185;
                    section.data(57).dtTransOffset = 164;

                    ;% rtP.Internal_B_jc_ebzizneqnl
                    section.data(58).logicalSrcIdx = 186;
                    section.data(58).dtTransOffset = 165;

                    ;% rtP.Internal_C_ir_hj1uxzp1ep
                    section.data(59).logicalSrcIdx = 187;
                    section.data(59).dtTransOffset = 167;

                    ;% rtP.Internal_C_jc_cs0yy5eb3e
                    section.data(60).logicalSrcIdx = 188;
                    section.data(60).dtTransOffset = 168;

                    ;% rtP.Internal_A_ir_oujtltlxgd
                    section.data(61).logicalSrcIdx = 189;
                    section.data(61).dtTransOffset = 172;

                    ;% rtP.Internal_A_jc_it04zokaos
                    section.data(62).logicalSrcIdx = 190;
                    section.data(62).dtTransOffset = 179;

                    ;% rtP.Internal_B_ir_ks0p0yho3p
                    section.data(63).logicalSrcIdx = 191;
                    section.data(63).dtTransOffset = 183;

                    ;% rtP.Internal_B_jc_cecpce5hwt
                    section.data(64).logicalSrcIdx = 192;
                    section.data(64).dtTransOffset = 184;

                    ;% rtP.Internal_C_ir_d33tkca5cs
                    section.data(65).logicalSrcIdx = 193;
                    section.data(65).dtTransOffset = 186;

                    ;% rtP.Internal_C_jc_hhgn2nuvph
                    section.data(66).logicalSrcIdx = 194;
                    section.data(66).dtTransOffset = 187;

                    ;% rtP.Internal_A_ir_oik1appi1t
                    section.data(67).logicalSrcIdx = 195;
                    section.data(67).dtTransOffset = 191;

                    ;% rtP.Internal_A_jc_dt4g4ekvp3
                    section.data(68).logicalSrcIdx = 196;
                    section.data(68).dtTransOffset = 192;

                    ;% rtP.Internal_B_ir_ov1s2pbumi
                    section.data(69).logicalSrcIdx = 197;
                    section.data(69).dtTransOffset = 194;

                    ;% rtP.Internal_B_jc_h12ogkfonh
                    section.data(70).logicalSrcIdx = 198;
                    section.data(70).dtTransOffset = 195;

                    ;% rtP.Internal_C_ir_pfmozql5e3
                    section.data(71).logicalSrcIdx = 199;
                    section.data(71).dtTransOffset = 197;

                    ;% rtP.Internal_C_jc_f3vgqu5len
                    section.data(72).logicalSrcIdx = 200;
                    section.data(72).dtTransOffset = 198;

            nTotData = nTotData + section.nData;
            paramMap.sections(2) = section;
            clear section


            ;%
            ;% Non-auto Data (parameter)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        paramMap.nTotData = nTotData;



    ;%**************************
    ;% Create Block Output Map *
    ;%**************************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 1;
        sectIdxOffset = 0;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc sigMap
        ;%
        sigMap.nSections           = nTotSects;
        sigMap.sectIdxOffset       = sectIdxOffset;
            sigMap.sections(nTotSects) = dumSection; %prealloc
        sigMap.nTotData            = -1;

        ;%
        ;% Auto data (rtB)
        ;%
            section.nData     = 20;
            section.data(20)  = dumData; %prealloc

                    ;% rtB.glp1tbzcj5
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

                    ;% rtB.do2ggtt1yj
                    section.data(2).logicalSrcIdx = 1;
                    section.data(2).dtTransOffset = 1;

                    ;% rtB.hkfwlkn44z
                    section.data(3).logicalSrcIdx = 2;
                    section.data(3).dtTransOffset = 2;

                    ;% rtB.k2ovkquskq
                    section.data(4).logicalSrcIdx = 3;
                    section.data(4).dtTransOffset = 3;

                    ;% rtB.ifrgwg0hqk
                    section.data(5).logicalSrcIdx = 4;
                    section.data(5).dtTransOffset = 4;

                    ;% rtB.cllkm5kpwc
                    section.data(6).logicalSrcIdx = 5;
                    section.data(6).dtTransOffset = 5;

                    ;% rtB.eu4yxwxe0f
                    section.data(7).logicalSrcIdx = 6;
                    section.data(7).dtTransOffset = 6;

                    ;% rtB.lqothvc25a
                    section.data(8).logicalSrcIdx = 7;
                    section.data(8).dtTransOffset = 7;

                    ;% rtB.j023pd3utu
                    section.data(9).logicalSrcIdx = 8;
                    section.data(9).dtTransOffset = 8;

                    ;% rtB.hlohoxodx1
                    section.data(10).logicalSrcIdx = 9;
                    section.data(10).dtTransOffset = 9;

                    ;% rtB.i5issijamp
                    section.data(11).logicalSrcIdx = 10;
                    section.data(11).dtTransOffset = 10;

                    ;% rtB.hrox2xbs04
                    section.data(12).logicalSrcIdx = 11;
                    section.data(12).dtTransOffset = 11;

                    ;% rtB.j24ks1lf4w
                    section.data(13).logicalSrcIdx = 12;
                    section.data(13).dtTransOffset = 12;

                    ;% rtB.gkhsfxl1z2
                    section.data(14).logicalSrcIdx = 13;
                    section.data(14).dtTransOffset = 13;

                    ;% rtB.gg2ss2zf1k
                    section.data(15).logicalSrcIdx = 14;
                    section.data(15).dtTransOffset = 14;

                    ;% rtB.duxrmorn55
                    section.data(16).logicalSrcIdx = 15;
                    section.data(16).dtTransOffset = 15;

                    ;% rtB.odngtpipys
                    section.data(17).logicalSrcIdx = 16;
                    section.data(17).dtTransOffset = 16;

                    ;% rtB.pojf1hfliu
                    section.data(18).logicalSrcIdx = 17;
                    section.data(18).dtTransOffset = 17;

                    ;% rtB.p4oo54b4y4
                    section.data(19).logicalSrcIdx = 18;
                    section.data(19).dtTransOffset = 18;

                    ;% rtB.hbqjo003ri
                    section.data(20).logicalSrcIdx = 19;
                    section.data(20).dtTransOffset = 19;

            nTotData = nTotData + section.nData;
            sigMap.sections(1) = section;
            clear section


            ;%
            ;% Non-auto Data (signal)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        sigMap.nTotData = nTotData;



    ;%*******************
    ;% Create DWork Map *
    ;%*******************
    
        nTotData      = 0; %add to this count as we go
        nTotSects     = 2;
        sectIdxOffset = 1;

        ;%
        ;% Define dummy sections & preallocate arrays
        ;%
        dumSection.nData = -1;
        dumSection.data  = [];

        dumData.logicalSrcIdx = -1;
        dumData.dtTransOffset = -1;

        ;%
        ;% Init/prealloc dworkMap
        ;%
        dworkMap.nSections           = nTotSects;
        dworkMap.sectIdxOffset       = sectIdxOffset;
            dworkMap.sections(nTotSects) = dumSection; %prealloc
        dworkMap.nTotData            = -1;

        ;%
        ;% Auto data (rtDW)
        ;%
            section.nData     = 24;
            section.data(24)  = dumData; %prealloc

                    ;% rtDW.kurh2j4r11.AQHandles
                    section.data(1).logicalSrcIdx = 0;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.mo0b2dlk3c.AQHandles
                    section.data(2).logicalSrcIdx = 1;
                    section.data(2).dtTransOffset = 1;

                    ;% rtDW.hhjg0tswq3.AQHandles
                    section.data(3).logicalSrcIdx = 2;
                    section.data(3).dtTransOffset = 2;

                    ;% rtDW.jwlvqglcx5.AQHandles
                    section.data(4).logicalSrcIdx = 3;
                    section.data(4).dtTransOffset = 3;

                    ;% rtDW.pexlwytvcb.AQHandles
                    section.data(5).logicalSrcIdx = 4;
                    section.data(5).dtTransOffset = 4;

                    ;% rtDW.ks21q1oe0m.AQHandles
                    section.data(6).logicalSrcIdx = 5;
                    section.data(6).dtTransOffset = 5;

                    ;% rtDW.kquv5edfh5.TimePtr
                    section.data(7).logicalSrcIdx = 6;
                    section.data(7).dtTransOffset = 6;

                    ;% rtDW.ck2c4rbpy4.TimePtr
                    section.data(8).logicalSrcIdx = 7;
                    section.data(8).dtTransOffset = 7;

                    ;% rtDW.jn41mxwh4n.TimePtr
                    section.data(9).logicalSrcIdx = 8;
                    section.data(9).dtTransOffset = 8;

                    ;% rtDW.l00cdg23mz.TimePtr
                    section.data(10).logicalSrcIdx = 9;
                    section.data(10).dtTransOffset = 9;

                    ;% rtDW.ijhh2juliq.TimePtr
                    section.data(11).logicalSrcIdx = 10;
                    section.data(11).dtTransOffset = 10;

                    ;% rtDW.kf2drr35ex.TimePtr
                    section.data(12).logicalSrcIdx = 11;
                    section.data(12).dtTransOffset = 11;

                    ;% rtDW.kvwfqgfdli.TimePtr
                    section.data(13).logicalSrcIdx = 12;
                    section.data(13).dtTransOffset = 12;

                    ;% rtDW.pt4k0lry0r.TimePtr
                    section.data(14).logicalSrcIdx = 13;
                    section.data(14).dtTransOffset = 13;

                    ;% rtDW.ccqit54xab.TimePtr
                    section.data(15).logicalSrcIdx = 14;
                    section.data(15).dtTransOffset = 14;

                    ;% rtDW.oxs2h0xie1.TimePtr
                    section.data(16).logicalSrcIdx = 15;
                    section.data(16).dtTransOffset = 15;

                    ;% rtDW.pgpv2tqduz.TimePtr
                    section.data(17).logicalSrcIdx = 16;
                    section.data(17).dtTransOffset = 16;

                    ;% rtDW.igxx1lod55.TimePtr
                    section.data(18).logicalSrcIdx = 17;
                    section.data(18).dtTransOffset = 17;

                    ;% rtDW.fmgvpmawvu.TimePtr
                    section.data(19).logicalSrcIdx = 18;
                    section.data(19).dtTransOffset = 18;

                    ;% rtDW.pfyaolnavt.TimePtr
                    section.data(20).logicalSrcIdx = 19;
                    section.data(20).dtTransOffset = 19;

                    ;% rtDW.lzv3r0nmzz.TimePtr
                    section.data(21).logicalSrcIdx = 20;
                    section.data(21).dtTransOffset = 20;

                    ;% rtDW.defpspt1xf.TimePtr
                    section.data(22).logicalSrcIdx = 21;
                    section.data(22).dtTransOffset = 21;

                    ;% rtDW.ayyzzi4kcp.TimePtr
                    section.data(23).logicalSrcIdx = 22;
                    section.data(23).dtTransOffset = 22;

                    ;% rtDW.kccsld1twl.TimePtr
                    section.data(24).logicalSrcIdx = 23;
                    section.data(24).dtTransOffset = 23;

            nTotData = nTotData + section.nData;
            dworkMap.sections(1) = section;
            clear section

            section.nData     = 18;
            section.data(18)  = dumData; %prealloc

                    ;% rtDW.ludljk1mkt.PrevIndex
                    section.data(1).logicalSrcIdx = 24;
                    section.data(1).dtTransOffset = 0;

                    ;% rtDW.mimikdlvla.PrevIndex
                    section.data(2).logicalSrcIdx = 25;
                    section.data(2).dtTransOffset = 1;

                    ;% rtDW.azbmwzhuxe.PrevIndex
                    section.data(3).logicalSrcIdx = 26;
                    section.data(3).dtTransOffset = 2;

                    ;% rtDW.j5a3pxorvd.PrevIndex
                    section.data(4).logicalSrcIdx = 27;
                    section.data(4).dtTransOffset = 3;

                    ;% rtDW.mpjornappt.PrevIndex
                    section.data(5).logicalSrcIdx = 28;
                    section.data(5).dtTransOffset = 4;

                    ;% rtDW.h1xxcnwaw1.PrevIndex
                    section.data(6).logicalSrcIdx = 29;
                    section.data(6).dtTransOffset = 5;

                    ;% rtDW.ayogcju54x.PrevIndex
                    section.data(7).logicalSrcIdx = 30;
                    section.data(7).dtTransOffset = 6;

                    ;% rtDW.a2dupmilus.PrevIndex
                    section.data(8).logicalSrcIdx = 31;
                    section.data(8).dtTransOffset = 7;

                    ;% rtDW.pjtji2hhge.PrevIndex
                    section.data(9).logicalSrcIdx = 32;
                    section.data(9).dtTransOffset = 8;

                    ;% rtDW.iicsswm23u.PrevIndex
                    section.data(10).logicalSrcIdx = 33;
                    section.data(10).dtTransOffset = 9;

                    ;% rtDW.cfr12oq5te.PrevIndex
                    section.data(11).logicalSrcIdx = 34;
                    section.data(11).dtTransOffset = 10;

                    ;% rtDW.m3yxmkxthp.PrevIndex
                    section.data(12).logicalSrcIdx = 35;
                    section.data(12).dtTransOffset = 11;

                    ;% rtDW.oevdd11tc0.PrevIndex
                    section.data(13).logicalSrcIdx = 36;
                    section.data(13).dtTransOffset = 12;

                    ;% rtDW.l5nxfgntsh.PrevIndex
                    section.data(14).logicalSrcIdx = 37;
                    section.data(14).dtTransOffset = 13;

                    ;% rtDW.j3zy2npopo.PrevIndex
                    section.data(15).logicalSrcIdx = 38;
                    section.data(15).dtTransOffset = 14;

                    ;% rtDW.hfvzo3ediv.PrevIndex
                    section.data(16).logicalSrcIdx = 39;
                    section.data(16).dtTransOffset = 15;

                    ;% rtDW.lxcs24fexb.PrevIndex
                    section.data(17).logicalSrcIdx = 40;
                    section.data(17).dtTransOffset = 16;

                    ;% rtDW.m1rn3czb2l.PrevIndex
                    section.data(18).logicalSrcIdx = 41;
                    section.data(18).dtTransOffset = 17;

            nTotData = nTotData + section.nData;
            dworkMap.sections(2) = section;
            clear section


            ;%
            ;% Non-auto Data (dwork)
            ;%


        ;%
        ;% Add final counts to struct.
        ;%
        dworkMap.nTotData = nTotData;



    ;%
    ;% Add individual maps to base struct.
    ;%

    targMap.paramMap  = paramMap;
    targMap.signalMap = sigMap;
    targMap.dworkMap  = dworkMap;

    ;%
    ;% Add checksums to base struct.
    ;%


    targMap.checksum0 = 3958171288;
    targMap.checksum1 = 3966129138;
    targMap.checksum2 = 2919969549;
    targMap.checksum3 = 1931496219;

