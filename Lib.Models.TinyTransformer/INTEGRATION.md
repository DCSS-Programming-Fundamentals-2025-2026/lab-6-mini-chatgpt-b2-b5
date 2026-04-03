# TinyTransformer Інтеграційний Гайд

## Створення моделі

Найпростіший спосіб - через фабрику з одним параметром:

```
var factory = new TinyTransformerModelFactory();
var model = factory.Create(vocabSize: 1000);
```

Модель автоматично ініціалізує ваги випадковими значеннями.

Якщо потрібен deterministic result (наприклад, для тестів) то можна передати seed:

```
var model = factory.Create(vocabSize: 1000, seed: 42);
```

## Конфігурація

За замовчуванням використовуються такі значення:   
- EmbeddingSize: 16   
- HeadCount: 1   
- ContextSize: 8

Якщо дефолтні значення не підходять то можна вказати все явно:

```
var model = factory.Create(
    vocabSize: 1000,
    embeddingSize: 32,
    headCount: 2,
    contextSize: 16,
    seed: 42
);
```

Model automatically truncates context to only `ContextSize` last tokens if the input sequence is larger than the available context window.

## Формат checkpoint

При серіалізації `ToPayload()` модель зберігається у такому форматі:

```
{
  "config": {
    "vocabSize": 1000,
    "embeddingSize": 16,
    "headCount": 1,
    "contextSize": 8
  },
  "tokenEmbeddings": [[...], [...]],
  "wq": [[...]],
  "wk": [[...]],
  "wv": [[...]],
  "wo": [[...]],
  "ffn1": [[...]],
  "ffn1Bias": [...],
  "ffn2": [[...]],
  "ffn2Bias": [...],
  "outputW": [[...]],
  "outputBias": [...]
}
```

Щоб зберегти та відновити модель:

```
// Зберігання
var payload = model.ToPayload();
string json = JsonSerializer.Serialize(payload);
File.WriteAllText("model.json", json);

// Відновлення
string json = File.ReadAllText("model.json");
using var doc = JsonDocument.Parse(json);
var restoredModel = factory.CreateFromPayload(doc.RootElement);
```

Після завантаження модель видає ідентичні результати - це перевірено інтеграційними тестами.