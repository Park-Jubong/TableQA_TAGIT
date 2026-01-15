<h1>TAGIT(TableQA with Code Generation and Iterative Prompting)</h1>

<p>
이 프로젝트는 <strong>WikiTableQuestions (wikitablequestions)</strong> 데이터셋을 활용해
테이블 기반 질의응답(Table Question Answering) 실험을 수행하는 파이프라인 예시입니다.
WikiTableQuestions는 위키피디아의 반정형(HTML) 테이블을 바탕으로 질문에 답하는 태스크를 다루며,
테스트 테이블이 학습 테이블과 분리되어 일반화 성능 평가에 자주 사용됩니다.
</p>

<hr/>

<h2>사용 데이터셋: WikiTableQuestions (wikitablequestions)</h2>

<ul>
  <li>
    공식 리포지토리:
    <a href="https://github.com/ppasupat/WikiTableQuestions" target="_blank" rel="noreferrer">
      ppasupat/WikiTableQuestions
    </a>
  </li>
  <li>
    프로젝트/설명 페이지:
    <a href="https://ppasupat.github.io/WikiTableQuestions/" target="_blank" rel="noreferrer">
      WikiTableQuestions 페이지
    </a>
  </li>
  <li>
    Hugging Face Datasets:
    <a href="https://huggingface.co/datasets/stanfordnlp/wikitablequestions" target="_blank" rel="noreferrer">
      stanfordnlp/wikitablequestions
    </a>
  </li>
  <li>
    TensorFlow Datasets:
    <a href="https://www.tensorflow.org/datasets/catalog/wiki_table_questions" target="_blank" rel="noreferrer">
      wiki_table_questions
    </a>
  </li>
  <li>
    논문(arXiv):
    <a href="https://arxiv.org/pdf/1508.00305" target="_blank" rel="noreferrer">
      Compositional Semantic Parsing on Semi-Structured Tables (Pasupat &amp; Liang, 2015)
    </a>
  </li>
</ul>

<p>
<strong>중요:</strong> 본 파이프라인의 입력(<code>--input_data</code>)은 WikiTableQuestions 데이터셋으로부터
전처리/가공된 형태여야 합니다. 예시 커맨드에서는 <code>table_sim_test</code>를 입력 데이터로 사용합니다.
</p>

<hr/>

<h2>파이프라인 개요</h2>

<ol>
  <li><strong>Prompt 생성</strong>: 데이터(테이블/질문)에서 LLM 입력 프롬프트를 구성</li>
  <li><strong>LLM 호출</strong>: 생성된 프롬프트로 모델 추론(샘플링 포함 가능)</li>
  <li><strong>정답 채점</strong>: 예측 결과와 정답 파일을 비교하여 성능 평가</li>
</ol>

<hr/>

<h2>실행 방법</h2>

<p>아래 커맨드는 사용자가 제공한 실행 순서를 그대로 정리한 것입니다.</p>

<h3>1) Prompt 생성</h3>
<pre><code class="language-bash">echo "1. generate prompt"
python generate_prompt.py \
  --prompt_type pyagent \
  --n_gram 1 \
  --input_data table_sim_test \
  --few_shot_num 1
</code></pre>

<h3>2) LLM 호출</h3>
<pre><code class="language-bash">echo "2. ask_llm"
python ask_llm.py \
  --prompt_type pyagent \
  --num_samples 100 \
  --output_file_name pyagent_test
</code></pre>

<h3>3) 정답 채점</h3>
<pre><code class="language-bash">echo "3. check answer"
python check_answer.py \
  --prediction_file pyagent_test \
  --answers_file pyagent_test
</code></pre>

<p>
<strong>주의:</strong> 일반적으로 <code>--answers_file</code>은 <em>정답(ground-truth)</em> 파일을 가리켜야 합니다.
현재 예시에서는 <code>--prediction_file</code>과 동일하게 <code>pyagent_test</code>가 들어가 있으므로,
실제 사용 시에는 WikiTableQuestions에서 추출/전처리한 정답 파일 경로로 바꿔 주세요.
</p>

<hr/>

<h2>주요 인자 설명</h2>

<h3>generate_prompt.py</h3>
<ul>
  <li><code>--prompt_type</code>: 프롬프트 템플릿 타입 (예: <code>pyagent</code>)</li>
  <li><code>--n_gram</code>: n-gram 설정(전처리/검색/유사도 로직에 사용되는 경우가 많음)</li>
  <li><code>--input_data</code>: 입력 데이터 이름/경로 (WikiTableQuestions 기반 전처리 결과여야 함)</li>
  <li><code>--few_shot_num</code>: few-shot 예시 개수</li>
</ul>

<h3>ask_llm.py</h3>
<ul>
  <li><code>--prompt_type</code>: 프롬프트 템플릿 타입</li>
  <li><code>--num_samples</code>: 샘플링/실행할 데이터 수 또는 샘플 수(구현에 따라 의미가 다를 수 있음)</li>
  <li><code>--output_file_name</code>: 예측 결과 저장 파일명(prefix)</li>
</ul>

<h3>check_answer.py</h3>
<ul>
  <li><code>--prediction_file</code>: 모델 예측 결과 파일</li>
  <li><code>--answers_file</code>: 정답 파일(WikiTableQuestions에서 생성한 ground-truth)</li>
</ul>

<hr/>

<h2>권장 디렉토리 구조 (예시)</h2>

<pre><code>project-root/
  ├─ generate_prompt.py
  ├─ ask_llm.py
  ├─ check_answer.py
  ├─ data/
  │   └─ wikitablequestions/        # WikiTableQuestions 원본/또는 전처리 데이터
  ├─ inputs/
  │   └─ table_sim_test/            # --input_data 로 사용하는 전처리 결과(예시)
  └─ outputs/
      └─ pyagent_test*              # 예측/로그/중간 산출물(구현에 따라 파일명 상이)
</code></pre>


<hr/>

<h2>Troubleshooting</h2>

<ul>
  <li><strong>입력 데이터 오류</strong>: <code>--input_data</code>가 WikiTableQuestions 기반 전처리 결과인지 확인하세요.</li>
  <li><strong>정답 채점 실패</strong>: <code>--answers_file</code>이 예측 파일이 아닌 정답 파일을 가리키는지 확인하세요.</li>
  <li><strong>재현성</strong>: 샘플링을 사용한다면 seed 고정 옵션(있다면)을 함께 사용하세요.</li>
</ul>
