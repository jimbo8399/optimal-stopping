echo "SVR with Linear Kernel"
./sim_d1_script_lin_svr.sh &> output.txt

echo "SVR with RBF Kernel"
./sim_d1_script_rbf_svr.sh &>> output.txt

echo "Linear Regression"
./sim_d2_script.sh &>> output.txt
