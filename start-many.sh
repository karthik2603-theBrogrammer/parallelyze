#!/bin/bash

chmod +x *.py
echo "{" > results.json

for np in 3 4 5 6 7 8
do
    echo "Running MPI program with $np processes..."
    
    # Run the MPI program and capture the output
    output=$(mpirun -np $np python3 many-mpi.py)
    
    # Extract relevant information from the output
    weights=$(echo "$output" | awk '/Averaged Weights:/ {print $3}')
    bias=$(echo "$output" | awk '/Averaged Biases:/ {print $3}')
    time=$(echo "$output" | awk '/Average Time/ {print $5}')
    mse=$(echo "$output" | awk '/Mean Squared Error/ {print $5}')
    
    # Remove square brackets from weights
    weights=${weights//[}
    weights=${weights//]}
    
    # Append the results to the JSON file
    echo "\"$np\": {" >> results.json
    echo "    \"Averaged Weights\": $weights," >> results.json
    echo "    \"Averaged Biases\": $bias," >> results.json
    echo "    \"time\": $time," >> results.json
    echo "    \"mse\": $mse" >> results.json

    if [ "$np" -ne 8 ]; then
        echo "}," >> results.json
    else
        echo "}" >> results.json
    fi
done

# Remove the trailing comma from the last entry
sed -i '$ s/,$//' results.json

# Close the JSON object
echo "}" >> results.json

python3 plot.py
