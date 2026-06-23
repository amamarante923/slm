import unittest
from unittest.mock import patch, MagicMock

from finetune import (
    carregar_tokenizador,
    aplicar_lora,
    carregar_dataset,
    criar_argumentos_treino,
    salvar_modelo,
)


class TestCarregarTokenizador(unittest.TestCase):
    """Testes para a função carregar_tokenizador."""

    @patch("finetune.AutoTokenizer")
    def test_retorna_tokenizador(self, mock_tokenizer: MagicMock) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = "token"
        mock_tokenizer.from_pretrained.return_value = mock_tok

        resultado = carregar_tokenizador("modelo")

        mock_tokenizer.from_pretrained.assert_called_once_with(
            "modelo", trust_remote_code=True
        )
        self.assertEqual(resultado, mock_tok)

    @patch("finetune.AutoTokenizer")
    def test_define_pad_token_quando_none(self, mock_tokenizer: MagicMock) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tok

        resultado = carregar_tokenizador("modelo")

        self.assertEqual(resultado.pad_token, "<eos>")

    @patch("finetune.AutoTokenizer")
    def test_mantem_pad_token_existente(self, mock_tokenizer: MagicMock) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tokenizer.from_pretrained.return_value = mock_tok

        resultado = carregar_tokenizador("modelo")

        self.assertEqual(resultado.pad_token, "<pad>")


class TestAplicarLora(unittest.TestCase):
    """Testes para a função aplicar_lora."""

    @patch("finetune.get_peft_model")
    @patch("finetune.LoraConfig")
    def test_aplica_lora_com_parametros_corretos(
        self, mock_config: MagicMock, mock_get_peft: MagicMock
    ) -> None:
        mock_modelo = MagicMock()
        mock_peft_modelo = MagicMock()
        mock_get_peft.return_value = mock_peft_modelo

        resultado = aplicar_lora(mock_modelo)

        mock_config.assert_called_once_with(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        mock_get_peft.assert_called_once_with(mock_modelo, mock_config.return_value)
        self.assertEqual(resultado, mock_peft_modelo)


class TestCarregarDataset(unittest.TestCase):
    """Testes para a função carregar_dataset."""

    @patch("finetune.load_dataset")
    def test_carrega_dataset_json(self, mock_load: MagicMock) -> None:
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        resultado = carregar_dataset("dados.jsonl")

        mock_load.assert_called_once_with(
            "json", data_files="dados.jsonl", split="train"
        )
        self.assertEqual(resultado, mock_dataset)


class TestCriarArgumentosTreino(unittest.TestCase):
    """Testes para a função criar_argumentos_treino."""

    @patch("finetune.SFTConfig")
    def test_cria_config_com_parametros(self, mock_config: MagicMock) -> None:
        criar_argumentos_treino("./saida", 5, 2, 1e-3)

        mock_config.assert_called_once_with(
            output_dir="./saida",
            num_train_epochs=5,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            logging_steps=1,
            save_strategy="epoch",
            fp16=False,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            use_cpu=False,
        )


class TestSalvarModelo(unittest.TestCase):
    """Testes para a função salvar_modelo."""

    def test_salva_modelo_e_tokenizador(self) -> None:
        mock_modelo = MagicMock()
        mock_tokenizador = MagicMock()

        salvar_modelo(mock_modelo, mock_tokenizador, "./saida")

        mock_modelo.save_pretrained.assert_called_once_with("./saida")
        mock_tokenizador.save_pretrained.assert_called_once_with("./saida")


if __name__ == "__main__":
    unittest.main()
