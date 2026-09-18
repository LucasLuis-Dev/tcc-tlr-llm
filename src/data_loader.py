import os
import glob
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import pandas as pd

# Caminhos padrão do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
REQUIREMENTS_DIR = RAW_DATA_DIR / "requirements"
TEST_CASES_DIR = RAW_DATA_DIR / "test_cases"
GROUND_TRUTH_FILE = RAW_DATA_DIR / "ground_truth.txt"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_CSV_FILE = PROCESSED_DATA_DIR / "itrust_processed.csv"


def load_requirements(requirements_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Lê todos os arquivos de requisitos (.txt) do diretório especificado.
    
    :param requirements_dir: Caminho para a pasta contendo os arquivos de requisitos.
    :return: Dicionário mapeando req_id (ex: 'UC10E1.txt') para o texto do requisito.
    """
    req_dir = Path(requirements_dir) if requirements_dir else REQUIREMENTS_DIR
    requirements = {}

    if not req_dir.exists():
        raise FileNotFoundError(f"Diretório de requisitos não encontrado: {req_dir}")

    for file_path in req_dir.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                requirements[file_path.name] = content
        except Exception as e:
            print(f"[Aviso] Falha ao ler requisito {file_path.name}: {e}")

    print(f"Total de requisitos carregados: {len(requirements)}")
    return requirements


def load_test_cases(test_cases_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Lê todos os arquivos de código/casos de teste (.java / .txt) do diretório especificado.
    
    :param test_cases_dir: Caminho para a pasta contendo os arquivos de casos de teste / classes.
    :return: Dicionário mapeando test_id (ex: 'GetUserNameAction.java') para o código/texto.
    """
    tc_dir = Path(test_cases_dir) if test_cases_dir else TEST_CASES_DIR
    test_cases = {}

    if not tc_dir.exists():
        raise FileNotFoundError(f"Diretório de casos de teste não encontrado: {tc_dir}")

    # Suporta arquivos .java e .txt, tanto no diretório plano quanto recursivo
    patterns = ["*.java", "*.txt", "**/*.java", "**/*.txt"]
    files = set()
    for pattern in patterns:
        files.update(tc_dir.glob(pattern))

    for file_path in files:
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    test_cases[file_path.name] = content
            except Exception as e:
                print(f"[Aviso] Falha ao ler caso de teste/código {file_path.name}: {e}")

    print(f"Total de casos de teste/classes carregados: {len(test_cases)}")
    return test_cases


def load_ground_truth(ground_truth_file: Optional[Path] = None) -> Set[Tuple[str, str]]:
    """
    Lê a matriz de rastreabilidade oficial (Gabarito / Ground Truth).
    Suporta formato padrão do benchmark iTrust (ex: 'UC10E1.txt: GetUserNameAction.java').
    
    :param ground_truth_file: Caminho para o arquivo de gabarito.
    :return: Conjunto de tuplas (req_id, test_id) que possuem elo verdadeiro.
    """
    gt_path = Path(ground_truth_file) if ground_truth_file else GROUND_TRUTH_FILE

    # Se não encontrar no caminho principal, tenta nome alternativo
    if not gt_path.exists():
        alt_path = RAW_DATA_DIR / "itrust_solution_links.txt"
        if alt_path.exists():
            gt_path = alt_path
        else:
            raise FileNotFoundError(f"Arquivo de gabarito não encontrado: {gt_path}")

    ground_truth_links = set()

    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Formato esperado: 'REQ_ID: TEST_ID' ou 'REQ_ID TEST_ID'
            if ":" in line:
                parts = line.split(":", 1)
                req_id = parts[0].strip()
                test_id = parts[1].strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    req_id = parts[0].strip()
                    test_id = parts[1].strip()
                else:
                    continue

            ground_truth_links.add((req_id, test_id))

    print(f"Total de elos válidos (ground truth = 1): {len(ground_truth_links)}")
    return ground_truth_links


def build_processed_dataset(
    requirements_dir: Optional[Path] = None,
    test_cases_dir: Optional[Path] = None,
    ground_truth_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Unifica requisitos, casos de teste e a matriz de rastreabilidade em um único DataFrame,
    gerando a matriz completa de candidatos (produto cartesiano).
    
    Exporta o DataFrame consolidado para 'data/processed/itrust_processed.csv'.
    
    :return: pd.DataFrame consolidado com colunas:
             [req_id, req_text, test_id, test_text, ground_truth]
    """
    out_file = Path(output_file) if output_file else PROCESSED_CSV_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Iniciando consolidação do Dataset iTrust ---")
    reqs = load_requirements(requirements_dir)
    tests = load_test_cases(test_cases_dir)
    ground_truth = load_ground_truth(ground_truth_file)

    rows = []
    print("Gerando pares de candidatos (requisitos x casos de teste)...")
    for req_id, req_text in reqs.items():
        for test_id, test_text in tests.items():
            is_linked = 1 if (req_id, test_id) in ground_truth else 0
            rows.append({
                "req_id": req_id,
                "req_text": req_text,
                "test_id": test_id,
                "test_text": test_text,
                "ground_truth": is_linked
            })

    df = pd.DataFrame(rows)
    print(f"Matriz consolidada com {len(df)} pares gerados.")
    print(f"Distribuição de classes: 1 (Positivos) = {df['ground_truth'].sum()} | 0 (Negativos) = {(df['ground_truth'] == 0).sum()}")

    # Exporta para CSV
    print(f"Salvando dataset consolidado em: {out_file}")
    df.to_csv(out_file, index=False, encoding="utf-8")
    print("Exportação concluída com sucesso!")

    return df


def load_itrust_dataset(sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Carrega o dataset consolidado do iTrust a partir de data/processed/itrust_processed.csv.
    Se o arquivo processado ainda não existir, constrói e salva automaticamente.
    
    :param sample_size: Se fornecido, retorna uma amostra balanceada ou aleatória de linhas.
    :param random_state: Semente para reprodutibilidade da amostragem.
    :return: DataFrame pronto para uso no pipeline de TLR.
    """
    if not PROCESSED_CSV_FILE.exists():
        print(f"Arquivo processado não encontrado em {PROCESSED_CSV_FILE}. Construindo agora...")
        df = build_processed_dataset()
    else:
        print(f"Carregando dataset processado de {PROCESSED_CSV_FILE}...")
        df = pd.read_csv(PROCESSED_CSV_FILE)

    if sample_size and sample_size < len(df):
        # Amostragem balanceada sempre que possível para garantir pares positivos na avaliação
        pos_df = df[df["ground_truth"] == 1]
        neg_df = df[df["ground_truth"] == 0]
        
        half_sample = sample_size // 2
        n_pos = min(len(pos_df), half_sample)
        n_neg = sample_size - n_pos

        sampled_pos = pos_df.sample(n=n_pos, random_state=random_state)
        sampled_neg = neg_df.sample(n=n_neg, random_state=random_state)
        sampled_df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"Amostra criada com {len(sampled_df)} pares ({n_pos} positivos, {n_neg} negativos).")
        return sampled_df

    return df


def get_mock_dataset() -> pd.DataFrame:
    """
    Mantido para testes rápidos de integração unitária sem ler o dataset de 100MB.
    """
    data = [
        {
            "req_id": "UC01",
            "req_text": "O sistema deve permitir que médicos visualizem o histórico de consultas e prescrições de seus pacientes.",
            "test_id": "TC01",
            "test_text": "Verificar se o médico autenticado consegue listar e visualizar as receitas anteriores do paciente selecionado.",
            "ground_truth": 1
        },
        {
            "req_id": "UC01",
            "req_text": "O sistema deve permitir que médicos visualizem o histórico de consultas e prescrições de seus pacientes.",
            "test_id": "TC08",
            "test_text": "Validar envio de e-mail de recuperação de senha quando o usuário esquece a senha de acesso.",
            "ground_truth": 0
        }
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Executa a consolidação e exportação ao rodar o script diretamente
    df_consolidated = build_processed_dataset()
    print("\nVisualização das primeiras linhas:")
    print(df_consolidated[["req_id", "test_id", "ground_truth"]].head())
