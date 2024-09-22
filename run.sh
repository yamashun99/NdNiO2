cd sampling_1e7
python analytic_continuation.py

cd ..
cd sampling_1e8
mpirun python dmft.py | tee dmft.out
