# src/prompt_templates.py


def get_zero_shot_prompt(req_text: str, test_text: str) -> str:
    """
    Retorna o template de prompt Zero-shot estruturado.
    A instrução de sistema define o papel da IA e a tarefa de forma objetiva.
    """
    prompt = f"""Você é um engenheiro de software sênior especialista em testes de software e validação de requisitos. 
Sua tarefa é identificar se existe um elo de rastreabilidade (Traceability Link) entre um Requisito de Software e um Caso de Teste.

Existe uma ligação direta e semântica entre eles? O caso de teste apresentado foi desenhado para validar o requisito apresentado?

[Requisito]:
{req_text}

[Caso de Teste]:
{test_text}

Responda APENAS com a palavra SIM ou NÃO."""
    return prompt


def get_few_shot_prompt(req_text: str, test_text: str) -> str:
    """
    Retorna o template de prompt Few-shot.
    Injeta exemplos prévios (um positivo e um negativo) extraídos do próprio domínio 
    para ensinar à IA o padrão de decisão esperado antes de avaliar o dado real.
    """
    prompt = f"""Você é um engenheiro de software sênior especialista em testes de software e validação de requisitos. 
Sua tarefa é identificar se existe um elo de rastreabilidade (Traceability Link) entre um Requisito de Software e um Caso de Teste. 
Abaixo estão dois exemplos de como você deve classificar, seguidos pelo caso real que você deve avaliar.

=== EXEMPLO 1 (COM LIGAÇÃO) ===
[Requisito]: O sistema deve permitir que o administrador registre um novo paciente com dados básicos.
[Caso de Teste]: public void testAddPatient() {{ assertNotNull(patientDAO.add(new Patient("John", "Doe"))); }}
Resposta: SIM

=== EXEMPLO 2 (SEM LIGAÇÃO) ===
[Requisito]: O sistema deve exibir o histórico de prescrições do paciente na tela principal.
[Caso de Teste]: public void testDeleteUser() {{ assertTrue(userDAO.delete(user.getId())); }}
Resposta: NÃO

=== CASO REAL A AVALIAR ===
Existe uma ligação direta e semântica entre o requisito e o teste abaixo? O teste foi desenhado para validar este requisito?

[Requisito]:
{req_text}

[Caso de Teste]:
{test_text}

Responda APENAS com a palavra SIM ou NÃO."""
    return prompt


def get_cot_prompt(req_text: str, test_text: str) -> str:
    """
    Retorna o template de prompt Chain-of-Thought (CoT).
    Instrui o modelo a raciocinar passo a passo antes de emitir a decisão final,
    concluindo com SIM ou NÃO na última linha.
    """
    prompt = f"""Você é um engenheiro de software sênior especialista em testes de software e validação de requisitos. 
Avalie se há ligação de rastreabilidade (Traceability Link) entre o Requisito e o Caso de Teste abaixo.

Passo 1: Analise os conceitos-chave e a funcionalidade esperada no requisito.
Passo 2: Analise o objetivo, as asserções e os artefatos do código do caso de teste.
Passo 3: Conclua se há validação direta entre o caso de teste e o requisito.

[Requisito]:
{req_text}

[Caso de Teste]:
{test_text}

Explique brevemente o raciocínio e, na última linha, responda APENAS com a palavra SIM ou NÃO."""
    return prompt

