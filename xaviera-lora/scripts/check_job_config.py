#!/usr/bin/env python3
"""
Check the configuration of a previous SageMaker training job to see what worked.
"""
import boto3
import json
import sys

def get_job_details(job_name):
    """Get detailed configuration of a SageMaker training job."""
    sagemaker = boto3.client('sagemaker')
    
    try:
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        
        print(f"\n=== Training Job: {job_name} ===")
        print(f"Status: {response['TrainingJobStatus']}")
        print(f"\nInstance Configuration:")
        print(f"  Instance Type: {response['ResourceConfig']['InstanceType']}")
        print(f"  Instance Count: {response['ResourceConfig']['InstanceCount']}")
        print(f"\nContainer Configuration:")
        print(f"  Algorithm: {response.get('AlgorithmSpecification', {}).get('TrainingInputMode', 'N/A')}")
        
        # Get container image
        containers = response.get('AlgorithmSpecification', {}).get('TrainingImage', 'N/A')
        if containers:
            print(f"  Container Image: {containers}")
        else:
            containers_list = response.get('AlgorithmSpecification', {}).get('ContainerEntrypoint', [])
            if containers_list:
                print(f"  Container Entrypoint: {containers_list}")
        
        # Get algorithm name (this is what triggers the validation)
        algorithm_name = response.get('AlgorithmSpecification', {}).get('AlgorithmName', 'N/A')
        training_input_mode = response.get('AlgorithmSpecification', {}).get('TrainingInputMode', 'N/A')
        print(f"  Algorithm Name: {algorithm_name}")
        print(f"  Training Input Mode: {training_input_mode}")
        
        # Check if there's any special configuration
        print(f"\nFull AlgorithmSpecification:")
        print(json.dumps(response.get('AlgorithmSpecification', {}), indent=2, default=str))
        
        # Get hyperparameters
        print(f"\nHyperparameters:")
        for key, value in response.get('HyperParameters', {}).items():
            print(f"  {key}: {value}")
        
        return response
        
    except Exception as e:
        print(f"Error getting job details: {e}")
        return None

if __name__ == "__main__":
    # Check a successful job from the history
    # Job: pytorch-training-2025-12-15-23-45-47-161 (Completed)
    job_name = "pytorch-training-2025-12-15-23-45-47-161"
    
    if len(sys.argv) > 1:
        job_name = sys.argv[1]
    
    print(f"Checking job: {job_name}")
    get_job_details(job_name)

