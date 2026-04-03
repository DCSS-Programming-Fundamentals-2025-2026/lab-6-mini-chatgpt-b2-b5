# B2: How to integrate - Training Data

Це опис, як у подальшому підключити/взаємодіяти із нашим проєктом

## Як створити IBatchProvider з масиву токенiв

Напряму це зробити не можна, тому алгоритм дій такий:

```
// готовий масив токенів
int[] tokens = { 0, 1, 2, 3, 1, 2, 3, 1, 0, 8 };

// цей масив надає ITokenizator на виході, що приймає наша частина загального проєкту

// створення потрібних екземплярів класу
ITokenStream stream = new ArrayTokenStream(tokens);
BatchWindowSampler sampler = new BatchWindowSampler();

// створення batch provider, що має метод GetBatch для отримання батчів
IBatchProvider batchProvider = new TokenBatchProvider(stream, sampler);

// задання параметрів для створення конфігурації батчів
int batchSize = 8 
int blockSize = 8192 

// створення конфігурації батчів
BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);
Random rng = new Random();

Batch batch = GetBatch(batchConfig, rng);
```

таким чином, отримуємо Batch -> це об'єкт, що зберігає множину батчів, що має:

int[][] Inputs -> це властивість, що відповідає за контекст
int[] Targets -> це властивість, що відповідає за ціль (що має видатись), використовується для навчання,
ціль є наступним токеном після контексту, що обирається "на рандом"

## Як створити TrainingConfig 

```
// задаємо параметри для конфігурації (значення є прикладом)
int epochs = 20
float learningRate = 0.01f
int checkpointInterval = 5

// створюємо об'єкт конфігурації
TrainingConfig trainingConfig = new TrainingConfig(epochs, learningRate, checkpointInterval)
```

## Як розпочати тренування моделі

```
ITrainingLoop trainingLoop = new TrainingLoop()

// виклик метода тренування
trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, tokens)
model -> це об'єкт типу ILanguageModel, у залежності від якого буде тренуватись конкретна модель
```

## Що є чим?

```
epochs -> кількість проходів по
batchSize -> кількість пар батчів
blockSize -> розмір контекстного вікна моделі
learningRate -> коефіцієнт сили оновлення ваг моделі
```

## Як це підключається до Trainer CLI 

1. CLI отримує токени
2. створює IBatchProvider
3. створює TrainingConfig
4. передає все у TrainingLoop

## Результат?

У результаті отримуємо тренування моделі



# B5: How to integrate - Neural

Це опис, як у подальшому підключити/взаємодіяти із нашим проєктом

## Як створити TinyNN із vocabSize

```
// розмір словника, що приходить від токенізатора
int vocabSize = tokenizer.VocabSize

// тип моделі
string modelKind = "tinynn"

// параметри моделі
int embeddingSize = 8192
int contextSize = 8

// створення конфігурації моделі + ваг
TinyNNConfig config = new TinyNNConfig(vocabSize, embeddingSize, contextSize)
TinyNNWeights weights = new TinyNNWeights(vocabSize, config)

// створення TinyNNModel
TinyNNModel model = new TinyNNModel(modelKind, vocabSize, config, weights)
```

## Що є чим?

EmbeddingSize -> це розмірність вектора, що представляє конкретний токен
ContextSize -> це розмір контекстного вікна моделі












