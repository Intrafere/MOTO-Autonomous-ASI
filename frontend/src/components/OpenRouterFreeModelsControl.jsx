import React from 'react';
import HelpTooltip from './HelpTooltip';

const FREE_MODELS_HELP_TEXT =
  'If you have more than $10 in your OpenRouter account, OpenRouter provides 1,000 free API calls per day that you can use with any program. These free models are most useful in support roles: as validators, the assistant, or supplemental brainstorm roles. They are typically not as knowledgeable as state-of-the-art models. Checking "Only show free OpenRouter models" filters the OpenRouter model lists to show only free models.';

export default function OpenRouterFreeModelsControl({
  checked,
  disabled = false,
  onChange,
}) {
  return (
    <span className="openrouter-free-models-control">
      <label className="openrouter-free-models-control__checkbox">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        Only show free OpenRouter models
      </label>
      <span className="openrouter-free-models-control__help">
        <span className="openrouter-free-models-control__help-label">Run MOTO for free</span>
        <HelpTooltip
          label="Run MOTO for free"
          buttonClassName="openrouter-free-models-control__help-button"
          popupClassName="openrouter-free-models-control__tooltip"
          useFixedPosition
          fixedPlacement="side-right"
        >
          {FREE_MODELS_HELP_TEXT}
        </HelpTooltip>
      </span>
      <span className="openrouter-free-models-control__help">
        <span className="openrouter-free-models-control__help-label">How to run MOTO like a pro</span>
        <HelpTooltip
          label="How to run MOTO like a pro"
          buttonClassName="openrouter-free-models-control__help-button"
          popupClassName="openrouter-free-models-control__tooltip openrouter-free-models-control__pro-tooltip"
          useFixedPosition
          fixedPlacement="side-right"
        >
          <span className="openrouter-free-models-control__pro-tips">
            <span className="openrouter-free-models-control__pro-section">
              <strong>Build useful proof memory</strong>
              <span>
                The Assistant retrieves useful previously verified proofs for the current run,
                so MOTO becomes more effective as you accumulate stored proofs or connect an
                available SyntheticLib corpus. It runs less often when it is not finding useful
                results and does not run when no proofs are available to retrieve.
              </span>
            </span>
            <span className="openrouter-free-models-control__pro-section">
              <strong>Spend stronger models where they matter</strong>
              <span>
                Reserve stronger or subscription-backed models, such as ChatGPT or Grok, for
                difficult roles like proof solving and writing. Submitters 2 and above, the
                Assistant, and often the Validator can use cheaper or free OpenRouter models.
              </span>
            </span>
            <span className="openrouter-free-models-control__pro-section">
              <strong>Control validation costs</strong>
              <span>
                Every rejected brainstorm still consumes a Validator call. Effective
                brainstormers reduce wasted calls, while an expensive Validator can
                significantly increase the total run cost.
              </span>
            </span>
            <span className="openrouter-free-models-control__pro-section">
              <strong>Tune proof solving</strong>
              <span>
                Prefer proof models marked with the PS tag. If a model repeatedly solves no
                proofs during a proof round, switch models or disable proof solving. When using
                an expensive proof model, Advanced Settings can limit concurrent proof attempts
                to one so MOTO works on one proof at a time.
              </span>
            </span>
          </span>
        </HelpTooltip>
      </span>
    </span>
  );
}
