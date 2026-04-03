using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Lib.Models.TinyNN.Tests")]
namespace Lib.Models.TinyNN.Layers
{
    public class LinearHead
    {
        public int VocabSize {get; set;}
        private TinyNNConfig _config;
        private TinyNNWeights _weights;

        public LinearHead(int vocabSize, TinyNNConfig tinyNNConfig, TinyNNWeights tinyNNWeights)
        {
            VocabSize = vocabSize;
            _config = tinyNNConfig;
            _weights = tinyNNWeights;
        }

        public float[] Project(float[] hidden)
        {
            float[] vector = MultiplyHiddenOnWeights(hidden);
            float[] logits = AddBiasToVector(vector);
            return logits; 
        }

        internal float[] AddBiasToVector(float[] vector)
        {
            float[] result = new float[vector.Length];
            for (int i = 0; i < _weights.OutputBias.Length; i++)
            {
                result[i] = vector[i] + _weights.OutputBias[i];
            }
            return result;
        }

        internal float[] MultiplyHiddenOnWeights(float[] hidden)
        {
            float[] vector = new float[_config.VocabSize];
            for(int i = 0; i < _config.VocabSize; i++)
            {
                for (int j = 0; j < hidden.Length; j++)
                {
                    vector[i] += hidden[j] * _weights.OutputWeights[j][i];
                }
            }

            return vector;
        }

        public float[] Backward(float[] hidden, float[] dLogits, float lr)
        {
            float[][] gradient = new float[_config.EmbeddingSize][];
            for (int i = 0; i < gradient.Length; i++)
                gradient[i] = new float[VocabSize];

            for (int i = 0; i < gradient[0].Length; i++)
            {
                for (int j = 0; j < gradient.Length; j++)
                {
                    gradient[j][i] = hidden[j] * dLogits[i];
                }      
            }

            for (int i = 0; i < _weights.OutputWeights.Length; i++)
            {
                for (int j = 0; j < _weights.OutputWeights[0].Length; j++)
                {
                    _weights.OutputWeights[i][j] -= gradient[i][j] * lr;
                }
            }

            for (int i = 0; i < _weights.OutputBias.Length; i++)
                _weights.OutputBias[i] -= lr * dLogits[i];

            float[] dHidden = new float[hidden.Length];
            for (int i = 0; i < _weights.OutputWeights.Length; i++)
            {
                float component = 0;
                for (int j = 0; j < _weights.OutputWeights[0].Length; j++)
                {
                    component += _weights.OutputWeights[i][j] * dLogits[j];    
                }

                dHidden[i] = component;
            }

            return dHidden;
        }
    }
}