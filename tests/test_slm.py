import unittest
from unittest.mock import patch, MagicMock

from slm import (
    _adaptador_valido,
    limpar_resposta,
    extrair_resposta,
    carregar_pipeline,
    carregar_pipeline_base,
    carregar_pipeline_com_adaptador,
)


class TestLimparResposta(unittest.TestCase):
    """Testes para a função limpar_resposta."""

    def test_remove_artefato_ground(self) -> None:
        texto = "Olá groundconteúdo indesejadoground mundo"
        resultado = limpar_resposta(texto)
        self.assertEqual(resultado, "Olá  mundo")

    def test_texto_sem_artefato(self) -> None:
        texto = "Texto limpo sem artefatos"
        resultado = limpar_resposta(texto)
        self.assertEqual(resultado, texto)

    def test_texto_vazio(self) -> None:
        resultado = limpar_resposta("")
        self.assertEqual(resultado, "")

    def test_multiplos_artefatos(self) -> None:
        texto = "A groundXground B groundYground C"
        resultado = limpar_resposta(texto)
        self.assertEqual(resultado, "A  B  C")

    def test_artefato_multilinhas(self) -> None:
        texto = "Antes ground\nlinha1\nlinha2\nground Depois"
        resultado = limpar_resposta(texto)
        self.assertEqual(resultado, "Antes  Depois")


class TestExtrairResposta(unittest.TestCase):
    """Testes para a função extrair_resposta."""

    def test_extrai_conteudo_corretamente(self) -> None:
        resultado = [
            {
                "generated_text": [
                    {"role": "user", "content": "pergunta"},
                    {"role": "assistant", "content": "resposta do modelo"},
                ]
            }
        ]
        self.assertEqual(extrair_resposta(resultado), "resposta do modelo")

    def test_extrai_resposta_vazia(self) -> None:
        resultado = [
            {
                "generated_text": [
                    {"role": "user", "content": "pergunta"},
                    {"role": "assistant", "content": ""},
                ]
            }
        ]
        self.assertEqual(extrair_resposta(resultado), "")


class TestCarregarPipeline(unittest.TestCase):
    """Testes para as funções de carregamento de pipeline."""

    @patch("slm._adaptador_valido", return_value=True)
    @patch("slm.carregar_pipeline_com_adaptador")
    def test_usa_adaptador_quando_existe(
        self, mock_com_adaptador: MagicMock, mock_valido: MagicMock
    ) -> None:
        mock_com_adaptador.return_value = MagicMock()
        carregar_pipeline("modelo", "./adaptador")
        mock_com_adaptador.assert_called_once_with("modelo", "./adaptador")

    @patch("slm._adaptador_valido", return_value=False)
    @patch("slm.carregar_pipeline_base")
    def test_usa_base_quando_adaptador_nao_existe(
        self, mock_base: MagicMock, mock_valido: MagicMock
    ) -> None:
        mock_base.return_value = MagicMock()
        carregar_pipeline("modelo", "./adaptador")
        mock_base.assert_called_once_with("modelo")

    @patch("slm._adaptador_valido", return_value=False)
    @patch("slm.carregar_pipeline_base")
    def test_usa_base_quando_adaptador_vazio(
        self, mock_base: MagicMock, mock_valido: MagicMock
    ) -> None:
        mock_base.return_value = MagicMock()
        carregar_pipeline("modelo", "./adaptador-vazio")
        mock_base.assert_called_once_with("modelo")

    @patch("slm._adaptador_valido", return_value=False)
    @patch("slm.carregar_pipeline_base")
    def test_usa_base_quando_adaptador_vazio(
        self, mock_base: MagicMock, mock_valido: MagicMock
    ) -> None:
        mock_base.return_value = MagicMock()
        carregar_pipeline("modelo", "./adaptador-vazio")
        mock_base.assert_called_once_with("modelo")

    @patch("slm.pipeline")
    def test_carregar_pipeline_base(self, mock_pipeline: MagicMock) -> None:
        mock_pipeline.return_value = MagicMock()
        carregar_pipeline_base("modelo-teste")
        mock_pipeline.assert_called_once_with(
            "text-generation", model="modelo-teste", max_new_tokens=2000
        )

    @patch("peft.PeftModel")
    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("slm.pipeline")
    def test_carregar_pipeline_com_adaptador(
        self,
        mock_pipeline: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_peft: MagicMock,
    ) -> None:
        mock_tok_inst = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok_inst
        mock_modelo_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_modelo_base
        mock_modelo_peft = MagicMock()
        mock_peft.from_pretrained.return_value = mock_modelo_peft

        carregar_pipeline_com_adaptador("modelo", "./adaptador")

        mock_tokenizer.from_pretrained.assert_called_once_with(
            "./adaptador", trust_remote_code=True
        )
        mock_peft.from_pretrained.assert_called_once_with(
            mock_modelo_base, "./adaptador"
        )
        mock_modelo_peft.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
