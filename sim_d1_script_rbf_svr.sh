echo "Applying Policy C"
python sim/sim_d1.py 25 R3 rbf policyC
echo "Done window 25"
###########################################
echo "Applying Policy E"
python sim/sim_d1.py 25 R3 rbf policyE
echo "Done window 25"
############################################
echo "Applying Policy A"
python sim/sim_d1.py 25 R3 rbf policyA
echo "Done window 25"
##########################################
echo "Applying Policy N"
python sim/sim_d1.py 25 R3 rbf policyN
echo "Done window 25"
###########################################
echo "Applying Policy M"
python sim/sim_d1.py 25 R3 rbf policyM
echo "Done window 25"
###########################################
echo "Applying Policy R"
python sim/sim_d1.py 25 R3 rbf policyR
echo "Done window 25"
###########################################
echo "Applying Policy OST"
python sim/sim_d1.py 25 R3 rbf policyOST 15
echo "Done window 25"
