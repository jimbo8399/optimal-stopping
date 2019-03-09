echo "Applying Policy C"
python sim/sim_d2.py 25 policyC
echo "Done window 25"

echo "Applying Policy A"
python sim/sim_d2.py 25 policyA
echo "Done window 25"

echo "Applying Policy E"
python sim/sim_d2.py 25 policyE
echo "Done window 25"

echo "Applying Policy R"
python sim/sim_d2.py 25 policyR
echo "Done window 25"

echo "Applying Policy M"
python sim/sim_d2.py 25 policyM
echo "Done window 25"

echo "Applying Policy N"
python sim/sim_d2.py 25 policyN
echo "Done window 25"

echo "Applying Policy OST"
python sim/sim_d2.py 25 policyOST 2
echo "Done windows 25"
