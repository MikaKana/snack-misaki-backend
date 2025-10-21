# Snack Misaki — Backend


※phase1ブランチにて、実装中

## 概要
このリポジトリは **Snack Misaki プロジェクトのバックエンド** です。
AWS Lambda (Python 3.11, Docker) をベースに、フロントエンドからの入力を処理し、  
小型 LLM や外部 LLM API を用いて応答を生成します。

- **Stage 2**: フロントエンドと連携し、小型 LLM (llama.cpp / GPT4All) で応答。Lambda 起動ごとにモデルを再利用するキャッシュ機構を備え、本番環境でも安定して動作します。
- **Stage 3**: 外部 LLM API (OpenAI / Bedrock / HuggingFace) と連携し高度な応答を実現

---

## 技術スタック
- **AWS Lambda (Python 3.11)** — サーバーレス基盤
- **Docker** — ローカル開発環境およびデプロイ用
- **llama.cpp / GPT4All** — 軽量なローカル推論モデル
- **OpenAI API / AWS Bedrock / HuggingFace Hub** — 外部 LLM API 連携

---

## 機能
1. **イベント受信**  
   フロントエンドから送信された入力を Lambda ハンドラで受け取る

2. **小型 LLM 呼び出し (Stage 2)**
   定型文で処理できない入力を llama.cpp / GPT4All で処理。構成ミスや推論エラーを検知した場合は自動的に Stage 3 の外部 LLM へフォールバックします。

3. **外部 API 呼び出し (Stage 3)**  
   小型 LLM で処理困難な入力は外部 LLM API にエスカレーション

4. **レスポンス返却**  
   JSON としてフロントエンドへ返す

---

## 実行方法（ローカル）
### Docker Compose (推奨)

開発時は Docker Compose で Lambda 互換の実行環境を立ち上げられます。

1. **初回セットアップ**
   ```bash
   docker compose up --build
   ```
   - `--build` を付けると依存パッケージを含むコンテナイメージが再構築されます。
   - バックエンドは `localhost:9000` (Lambda RIE の既定ポート) で待ち受けます。

2. **コード修正の反映**
   - ホストで変更したコードはコンテナにマウントされるため、コンテナを再起動しなくても即座に反映されます。
   - バックエンドのログは `docker compose logs -f` で確認できます。

3. **イベント送信（動作確認）**
   - Lambda RIE では [Invoke エンドポイント](https://docs.aws.amazon.com/ja_jp/lambda/latest/dg/images-test.html) に対してリクエストを送ります。
   - 例: `curl` で JSON ペイロードを送信する
     ```bash
     curl -X POST \
       "http://localhost:9000/2015-03-31/functions/function/invocations" \
       -H "Content-Type: application/json" \
       -d '{"input": "こんばんは"}'
     ```
   - 期待するレスポンスが返ってくるかを確認してください。

4. **テストの実行**
   - 開発コンテナ内でテストを走らせる場合（Lambda ランタイムのエントリポイントを無効化して pytest を実行）
     ```bash
     docker compose run --rm --entrypoint "" lambda python -m pytest
     ```
   - ホスト環境で直接テストする場合（PEP 621 形式の依存管理を使用）
     ```bash
     pip install .[dev]
     pytest
     ```

5. **Lint の実行**
   - 開発コンテナ内で実行する場合
     ```bash
     docker compose run --rm --entrypoint "" lambda python -m ruff check .
     ```
   - ホスト環境で直接実行する場合
     ```bash
     pip install .[dev]
     ruff check .
     ```

6. **自動整形（フォーマッタ）の実行**
   - 開発コンテナ内で実行する場合
     ```bash
     docker compose run --rm --entrypoint "" lambda python -m ruff format
     ```
   - ホスト環境で直接実行する場合
     ```bash
     pip install .[dev]
     ruff format
     ```

### Docker build
```bash
docker build -t snack-misaki-backend .
```

### Lambda エミュレータ起動
```bash
docker run -p 9000:8080 snack-misaki-backend
```

---

## 環境変数
`.env` または AWS Lambda の環境変数で設定します。

- `USE_LOCAL_LLM` : true の場合、小型 LLM (llama.cpp / GPT4All) を利用
- `LOCAL_LLM_BACKEND` : `gpt4all` / `llama.cpp` など利用するバックエンド (`auto` の場合は自動検出)
- `LOCAL_LLM_MODEL` : ローカル推論に利用するモデルファイル (GGUF/GPT4All)
- `LOCAL_LLM_MAX_TOKENS` : ローカルモデルの最大生成トークン数 (省略可、デフォルト256)
- `LOCAL_LLM_TEMPERATURE` : 生成時の温度パラメータ (省略可、デフォルト0.7)
- `LOCAL_LLM_ALLOW_FALLBACK` : true の場合のみ、ローカル LLM が利用できない際に定型文へフォールバック (デフォルト false)
- `OPENAI_API_KEY` : OpenAI API キー
- `BEDROCK_CREDENTIALS` : AWS Bedrock 用の認証情報
- `HUGGINGFACE_TOKEN` : HuggingFace Hub 用の認証トークン

---

## カスタマイズ方法
- **小型 LLM の切替**: llama.cpp / GPT4All を選択可能
- **外部 API の利用有無**: 環境変数で切替
- **応答ロジックの変更**: `app/handler.py` を編集

---

## 今後の拡張
- **認証/認可**: API Gateway + Cognito で保護
- **監視/ロギング**: CloudWatch との統合
- **CI/CD**: GitHub Actions から Lambda 自動デプロイ
- **個室モード連携**: ユーザーごとの会話を DynamoDB などに保存し、フロントと共有  

