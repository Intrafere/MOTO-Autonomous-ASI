import React from 'react';
import HelpTooltip from './HelpTooltip';

const FREE_MODELS_HELP_TEXT =
  'If you have more than $10 in your OpenRouter account, OpenRouter provides 1,000 free API calls per day that you can use with any program. These free models are useful as validators or supplemental brainstorm roles. They are typically not as knowledgeable as state-of-the-art models. Checking "Only show free OpenRouter models" filters the OpenRouter model lists to show only free models.';

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
    </span>
  );
}
