from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", dest="subscription_id", required=True)
    parser.add_argument("--resource-group", dest="resource_group", required=True)
    parser.add_argument("--workspace", dest="workspace", required=True)
    parser.add_argument("--endpoint-name", dest="endpoint_name", default="diabetes-endpoint")
    parser.add_argument("--deployment-name", dest="deployment_name", default="blue")
    return parser.parse_args()

def get_ml_client(subscription_id, resource_group, workspace):
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

def ensure_endpoint(ml_client, endpoint_name):
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
        return endpoint
    except Exception:
        print(f"Creating new endpoint '{endpoint_name}'...")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Online endpoint for MLflow diabetes model",
            auth_mode="key",
        )
        return ml_client.begin_create_or_update(endpoint).result()

def create_or_update_deployment(ml_client, endpoint_name, deployment_name):
    # Get latest registered model from Azure ML registry
    print("Fetching latest registered model 'diabetes-model'...")
    latest_model = ml_client.models.get(name="diabetes-model", label="latest")
    print(f"Using model: {latest_model.name} version {latest_model.version}")

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=latest_model.id,
        instance_type="Standard_D2as_v4",
        instance_count=1,
    )
    return ml_client.online_deployments.begin_create_or_update(deployment).result()

def set_traffic_to_deployment(ml_client, endpoint_name, deployment_name):
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.begin_create_or_update(endpoint).result()

def main():
    args = parse_args()
    print("Connecting to Azure Machine Learning workspace...")
    ml_client = get_ml_client(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    print(f"Ensuring online endpoint '{args.endpoint_name}' exists...")
    endpoint = ensure_endpoint(ml_client, args.endpoint_name)
    print(f"Using endpoint: {endpoint.name}")

    print(f"Creating or updating deployment '{args.deployment_name}'...")
    deployment = create_or_update_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint.name,
        deployment_name=args.deployment_name,
    )
    print(f"Deployment state: {deployment.provisioning_state}")

    print("Directing 100% of traffic to the deployment...")
    set_traffic_to_deployment(ml_client, endpoint.name, args.deployment_name)

    endpoint = ml_client.online_endpoints.get(name=endpoint.name)
    print(f"Deployment complete. Scoring URI: {endpoint.scoring_uri}")

if __name__ == "__main__":
    main()