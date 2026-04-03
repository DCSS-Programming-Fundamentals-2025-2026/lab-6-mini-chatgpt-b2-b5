namespace Lib.MathCore;

public static class MathOps
{
    public static IMathOps Default { get; } = new MathOpsImpl();
}
// MathOps.Default.ArgMax(...)