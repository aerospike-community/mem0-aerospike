from pydantic import BaseModel, ConfigDict, Field


class AerospikeConfig(BaseModel):
    """Configuration for the Aerospike vector store provider."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    namespace: str = Field("test", description="Aerospike namespace for the vector store")
    collection_name: str = Field("mem0", description="Aerospike set name for the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding vectors")
    host: str = Field("localhost", description="Aerospike seed host")
    port: int = Field(3000, description="Aerospike seed port")
    allow_scans_with_where: bool = Field(
        False,
        description="Allow filtered list/search queries that do not scope by user_id/agent_id/run_id to fall back to a scan",
    )
    max_record_bytes: int = Field(
        943718,
        description="Maximum serialized record size in bytes before client-side rejection (default ~0.9 MiB)",
    )
