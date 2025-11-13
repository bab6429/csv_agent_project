"""
API FastAPI pour l'agent CSV
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
from dotenv import load_dotenv
from csv_agent import CSVAgent

# Charger les variables d'environnement
load_dotenv()

app = FastAPI(
    title="CSV Agent API",
    description="API pour analyser des fichiers CSV avec l'IA",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis n'importe quelle origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage temporaire des agents (en production, utilisez une base de données ou un cache)
agents_store = {}


class QueryRequest(BaseModel):
    """Modèle pour les requêtes de questions"""
    question: str
    session_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    """Modèle pour les réponses"""
    answer: str
    session_id: str


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "CSV Agent API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Uploader un fichier CSV",
            "query": "POST /query - Poser une question sur les données",
            "health": "GET /health - Vérifier l'état de l'API"
        }
    }


@app.get("/health")
async def health():
    """Vérifier l'état de l'API"""
    api_key_configured = bool(os.getenv("GOOGLE_API_KEY"))
    return {
        "status": "healthy",
        "api_key_configured": api_key_configured,
        "active_sessions": len(agents_store)
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = "default",
    api_key: Optional[str] = None
):
    """
    Uploader un fichier CSV et initialiser l'agent
    
    Args:
        file: Fichier CSV/Excel à uploader
        session_id: Identifiant de session (optionnel)
        api_key: Clé API Google Gemini (optionnel, peut être dans .env)
    """
    try:
        # Vérifier que c'est un fichier CSV ou Excel
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Le fichier doit être au format CSV ou Excel (.csv, .xlsx, .xls)"
            )
        
        # Créer un fichier temporaire
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Créer l'agent
        try:
            agent = CSVAgent(
                tmp_path,
                api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                verbose=False
            )
            agents_store[session_id] = {
                "agent": agent,
                "file_path": tmp_path,
                "filename": file.filename
            }
        except Exception as e:
            # Nettoyer le fichier temporaire en cas d'erreur
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de l'initialisation de l'agent: {str(e)}"
            )
        
        return {
            "message": "Fichier uploadé avec succès",
            "session_id": session_id,
            "filename": file.filename,
            "file_size": len(content)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'upload: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Poser une question à l'agent sur les données CSV
    
    Args:
        request: Objet contenant la question et le session_id
    """
    if request.session_id not in agents_store:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' introuvable. Veuillez d'abord uploader un fichier CSV via /upload"
        )
    
    try:
        agent_data = agents_store[request.session_id]
        agent = agent_data["agent"]
        
        # Exécuter la requête
        answer = agent.query(request.question)
        
        return QueryResponse(
            answer=answer,
            session_id=request.session_id
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la question: {str(e)}"
        )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Supprimer une session et nettoyer les ressources
    
    Args:
        session_id: Identifiant de la session à supprimer
    """
    if session_id not in agents_store:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' introuvable"
        )
    
    try:
        agent_data = agents_store[session_id]
        # Nettoyer le fichier temporaire
        if os.path.exists(agent_data["file_path"]):
            os.unlink(agent_data["file_path"])
        
        # Supprimer de la store
        del agents_store[session_id]
        
        return {
            "message": f"Session '{session_id}' supprimée avec succès"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la suppression de la session: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

