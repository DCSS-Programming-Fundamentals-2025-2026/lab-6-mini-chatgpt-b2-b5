using Lib.Models.TinyNN.Configuration;
namespace Lib.Models.TinyNN.State
{
    public class TinyNNPayload
    {
        public TinyNNConfig Config {get; set;}
        public TinyNNWeights Weights {get; set;}
    }
}