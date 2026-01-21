## 🏗 Extended Architecture (MLOps & Data Lake)

The platform has been upgraded to a full **Lakehouse Architecture**:

```mermaid
graph LR
    subgraph Data_Lake ["🌊 Data Lake (MinIO)"]
        Raw[(Raw Zone)]
        Processed[(Processed Zone)]
        Artifacts[(Model Registry)]
    end

    subgraph Operations ["⚙️ MLOps Pipeline"]
        A[Ingestion Script] -->|Write| Raw
        B[Spark ETL] -->|Read| Raw
        B -->|Transform| B1{Clean & Validate}
        B1 -->|Write| Processed
        C[Training Job] -->|Read| Processed
        C -->|Track Metrics| D[MLflow Tracking]
        C -->|Save Model| Artifacts
    end

    subgraph Production ["🚀 Serving Layer"]
        API[FastAPI] -->|Load Model| Artifacts
        API -->|Predict| User
    end