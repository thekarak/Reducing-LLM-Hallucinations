import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data_loader import Document
from src.evaluator import compute_f1
from src.llm_client import LLMClient
from src.truthscope import analyze_claims, assess_risk, build_learning_coach
from src.vector_store import EmbeddingEngine, SimpleVectorStore


class EvaluatorTests(unittest.TestCase):
    def test_f1_counts_repeated_tokens(self):
        self.assertEqual(compute_f1("Mars Mars rover", "Mars rover"), 0.8)


class TruthScopeTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                "Curiosity uses a Multi-Mission Radioisotope Thermoelectric Generator. "
                "Juno uses three solar panel arrays.",
                {"source": "power_sources.txt", "chunk_id": "power_0"},
            )
        ]

    def test_claim_inspector_links_supported_claim_to_evidence(self):
        analysis = analyze_claims(
            "Curiosity uses a radioisotope thermoelectric generator.",
            self.documents,
            [0.82],
        )
        self.assertEqual(analysis["claims"][0]["status"], "supported")
        self.assertEqual(analysis["claims"][0]["evidence"]["source"], "power_sources.txt")
        self.assertEqual(analysis["summary"]["verified_claims"], 1)

    def test_negated_claim_is_not_marked_supported(self):
        analysis = analyze_claims(
            "Juno does not use three solar panel arrays.",
            self.documents,
            [0.82],
        )
        self.assertEqual(analysis["claims"][0]["status"], "unsupported")
        self.assertEqual(analysis["summary"]["citation_coverage_pct"], 0.0)

    def test_baseline_claims_are_unverified(self):
        analysis = analyze_claims("Curiosity uses nuclear power.")
        self.assertTrue(all(claim["status"] == "unverified" for claim in analysis["claims"]))

    def test_false_premise_and_missing_evidence_raise_risk(self):
        risk = assess_risk(
            "Did the fictional Apollo 18 rover definitively prove life?",
            [],
            [],
            "Out_Of_Corpus",
        )
        self.assertEqual(risk["level"], "High")
        self.assertGreaterEqual(risk["score"], 65)

    def test_learning_coach_prioritizes_unsupported_claim(self):
        analysis = analyze_claims("Curiosity returned 500 kilograms of dust.", self.documents, [0.2])
        risk = assess_risk("How many kilograms did Curiosity return?", self.documents, [0.2])
        coach = build_learning_coach(analysis, risk)
        self.assertIn("knowledge gap", coach["lesson"])
        self.assertTrue(coach["practice"])


class VectorStoreTests(unittest.TestCase):
    def make_tfidf_store(self):
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        engine.model_name = "test-tfidf"
        engine.model = None
        engine.is_neural = False
        engine.engine_type = "tfidf"
        engine._tfidf = None
        store = SimpleVectorStore(engine, similarity_threshold=0.0, retrieval_mode="dense")
        store.add_documents([
            Document("Curiosity uses radioisotope power.", {"source": "curiosity.txt"}),
            Document("Juno uses solar panels.", {"source": "juno.txt"}),
        ])
        return store

    def test_tfidf_vectorizer_survives_save_and_load(self):
        source = self.make_tfidf_store()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source.save(directory)
            loaded = SimpleVectorStore.load(
                directory,
                embedding_engine=source.embedding_engine,
                similarity_threshold=0.0,
                retrieval_mode="dense",
            )
            results = loaded.similarity_search_with_score("solar panels", k=1)
            self.assertEqual(results[0][0].metadata["source"], "juno.txt")
            self.assertIsNotNone(loaded.embedding_engine._tfidf)


class ProviderTests(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_openai_key_is_read_when_generation_runs(self, openai_client):
        completion = MagicMock()
        completion.choices[0].message.content = "Grounded response"
        openai_client.return_value.chat.completions.create.return_value = completion
        response = LLMClient(provider="openai", model="gpt-4o").generate("prompt")
        self.assertEqual(response, "Grounded response")
        openai_client.assert_called_once_with(api_key="test-key")

    @patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False)
    def test_missing_provider_key_records_simulator_fallback(self):
        client = LLMClient(provider="groq", model="llama-3.1-8b-instant")
        client.generate("Question: What happened on Apollo 11?")
        self.assertEqual(client.mock_fallback_calls, 1)


if __name__ == "__main__":
    unittest.main()
