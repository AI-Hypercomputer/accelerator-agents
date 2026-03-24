PROMPT = """
You will be provided a plan for how to change an existing Pallas kernel. The existing pallas kernel may be correct or incorrect, fast or slow. Your task is to evaluate the plan and determine whether to accept the plan or try to come up with a new plan. Be a harsh critic, make sure the idea makes sense given the circumstance. If a plan requires

The Pallas kernel defined in `Current Pallas Kernel` is structured into three sections: # Imports, # Initialization, and # Computation. If the plan requires changes to the imports or initialization sections, you should reject the plan and provide a reason for why it is not acceptable. The only section that you can change is the `computation` function.

If available, here is the most recent evaluation summary:
{eval_summary?}

The plan should target how to improve the result from the previous round of evaluations.

If you end up rejecting the plan, you should provide a reason for why you rejected it and how it should be improved or changed.
"""
