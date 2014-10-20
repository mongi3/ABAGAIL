package opt.example;

import opt.EvaluationFunction;
import shared.Instance;

/*
 * Use this class to wrap another evaluation function and get call counting capabilities
 */
public class CallCountingEvaluationFunction implements EvaluationFunction {
	public int callCnt = 0;
	private EvaluationFunction evalfun;
	
	public CallCountingEvaluationFunction(EvaluationFunction eval) {
		evalfun = eval;
	}
	
    public double value(Instance d) {
    	callCnt++;
    	return evalfun.value(d);
    }
    
    // Returns value, but does not increment count
    public double valueNoCount(Instance d) {
    	return evalfun.value(d);
    }
}
