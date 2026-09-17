import pandas as pd


def get_mock_dataset() -> pd.DataFrame:
    """
    Gera um DataFrame simulado com pares de Requisitos e Casos de Teste (estilo iTrust).
    Permite validar o fluxo de execução e métricas de TLR antes de carregar o dataset completo.
    
    Colunas:
        req_id (str): Identificador do requisito de software.
        req_text (str): Descrição textual do requisito.
        test_id (str): Identificador do caso de teste.
        test_text (str): Descrição textual do caso de teste.
        ground_truth (int): 1 se existe elo de rastreabilidade, 0 caso contrário.
    """
    data = [
        {
            "req_id": "UC01",
            "req_text": "O sistema deve permitir que profissionais de saúde autenticados acessem o histórico médico e prescrições de seus pacientes cadastrados.",
            "test_id": "TC01",
            "test_text": "Verificar se um médico autenticado consegue visualizar a lista de diagnósticos e receitas anteriores de um paciente específico.",
            "ground_truth": 1
        },
        {
            "req_id": "UC01",
            "req_text": "O sistema deve permitir que profissionais de saúde autenticados acessem o histórico médico e prescrições de seus pacientes cadastrados.",
            "test_id": "TC08",
            "test_text": "Validar se o envio de e-mail com token temporário de redefinição de senha ocorre corretamente após solicitação de esqueci minha senha.",
            "ground_truth": 0
        },
        {
            "req_id": "UC02",
            "req_text": "O sistema deve registrar trilhas de auditoria contendo timestamp, identificador do usuário e ação executada sempre que prontuários médicos forem editados.",
            "test_id": "TC05",
            "test_text": "Testar a inserção de novo registro no banco de dados de log de auditoria após a atualização de informações clínicas do prontuário.",
            "ground_truth": 1
        },
        {
            "req_id": "UC03",
            "req_text": "Pacientes podem agendar consultas ambulatoriais selecionando especialidade, profissional disponível e horário na agenda médica.",
            "test_id": "TC12",
            "test_text": "Verificar se o relatório contábil mensal calcula o total faturado por convênio de saúde aplicando as alíquotas de retenção de impostos.",
            "ground_truth": 0
        }
    ]

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = get_mock_dataset()
    print("--- Mock Dataset iTrust ---")
    print(df[["req_id", "test_id", "ground_truth"]])
