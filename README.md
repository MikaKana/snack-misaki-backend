# Snack Misaki — Backend

## 概要
このリポジトリは **Snack Misaki プロジェクトのバックエンド** です。  
AWS Lambda (Python 3.11, Docker) をベースに、フロントエンドからの入力を処理し、  
小型 LLM や外部 LLM API を用いて応答を生成します。

- **Stage 2**: フロントエンドと連携し、小型 LLM (llama.cpp / GPT4All) で応答
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
   定型文で処理できない入力を llama.cpp / GPT4All で処理

3. **外部 API 呼び出し (Stage 3)**  
   小型 LLM で処理困難な入力は外部 LLM API にエスカレーション

4. **レスポンス返却**  
   JSON としてフロントエンドへ返す

---

## 実行方法（ローカル）
### Docker build
```bash
docker build -t snack-misaki-backend .
```

### Lambda エミュレータ起動
```bash
docker run -p 9000:8080 snack-misaki-backend
```

### イベント送信
```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"input":"こんばんは"}'
```

---

## 環境変数
`.env` または AWS Lambda の環境変数で設定します。

- `USE_LOCAL_LLM` : true の場合、小型 LLM (llama.cpp / GPT4All) を利用
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

