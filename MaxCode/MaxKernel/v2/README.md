## Loading and Invoking the Skill Package (for Human Users)

To make MaxKernel available to your Jetski instance, you need to register
the skill in your user configuration.

### 1. Registering the Skills

Add the paths to the skills inside your user agent configuration file at:
`//depot/configs/users/<ldap>/_agents/skills.json`

You can either register the overall skill directory containing all sub-skills:

```json
"skill_directories": [
  "//depot/google3/third_party/py/accelerator_agents/MaxKernel/v2/"
]
```

### 2. Invoking the Skill

Once registered, the skill package is available to the agent.

*   **Automatic Activation**: The agent will automatically detect and load these
    skills when you ask it to optimize a Pallas TPU kernel.
*   **Explicit Invocation**: You can explicitly instruct the agent to use this
    framework by referencing the orchestrator skill name in your prompt: >
    "Using the `MaxKernel` skill, optimize the Pallas kernel
    in..."