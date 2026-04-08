"""Streamlit control room for Roitelet LLM.

The UI intentionally follows the product brief from the Roitelet page: a very
simple prompting interface, a lightweight left history column, and a separate
configuration and monitoring area.

Examples
--------
Run locally:
    streamlit run streamlit_app.py

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx
import pandas as pd
import requests
import streamlit as st

from app.config import get_settings
from app.schemas import AppSettingsPayload
from app.storage import storage

settings = get_settings()
API_BASE = settings.public_base_url.rstrip('/')


def require_login() -> None:
    """Display a simple login gate before showing the control room."""
    st.session_state.setdefault('authenticated', False)
    if st.session_state['authenticated']:
        return
    st.set_page_config(page_title='Roitelet LLM', layout='wide')
    st.title('🐦 Roitelet LLM')
    st.caption('Adaptive top-3 LLM routing with local synthesis.')
    with st.form('login'):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        submitted = st.form_submit_button('Log in')
    if submitted:
        if username == settings.admin_username and password == settings.admin_password:
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error('Invalid credentials.')
    st.stop()


def api_get(path: str) -> Any:
    """Perform a GET request against the local API."""
    response = requests.get(f'{API_BASE}{path}', timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: Dict[str, Any]) -> Any:
    """Perform a POST request against the local API."""
    response = requests.post(f'{API_BASE}{path}', json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def page_config() -> None:
    """Configure Streamlit page metadata and sidebar brand."""
    st.set_page_config(page_title='Roitelet LLM', page_icon='🐦', layout='wide')
    st.sidebar.title('🐦 Roitelet LLM')
    st.sidebar.markdown('[Roitelet project page](https://deraison.ai/en/roitelet)')
    st.sidebar.caption('Wing flap 1: discovery • flap 2: trio • flap 3: coronation')


def render_settings() -> None:
    """Render the configuration page."""
    st.header('Configuration')
    current = AppSettingsPayload.model_validate(api_get('/api/settings'))
    # Use a direct synchronous HTTP call (httpx) to list Ollama models.
    # The OllamaClient.list_models() is an async coroutine and cannot be
    # awaited or called in a synchronous Streamlit context.
    try:
        response = httpx.get(f"{current.ollama_base_url.rstrip('/')}/api/tags", timeout=5.0)
        response.raise_for_status()
        ollama_models: List[str] = [item['name'] for item in response.json().get('models', [])]
    except Exception:
        ollama_models = []
    with st.form('settings-form'):
        openrouter_api_key = st.text_input('OpenRouter API key', value=current.openrouter_api_key, type='password')
        openai_api_key = st.text_input('OpenAI API key', value=current.openai_api_key, type='password')
        anthropic_api_key = st.text_input('Anthropic API key', value=current.anthropic_api_key, type='password')
        gemini_api_key = st.text_input('Gemini API key', value=current.gemini_api_key, type='password')
        perplexity_api_key = st.text_input('Perplexity API key', value=current.perplexity_api_key, type='password')
        openai_compatible_api_key = st.text_input('OpenAI-compatible API key', value=current.openai_compatible_api_key, type='password')
        openai_compatible_base_url = st.text_input('OpenAI-compatible base URL', value=current.openai_compatible_base_url)
        openai_compatible_model = st.text_input('OpenAI-compatible default model', value=current.openai_compatible_model)
        ollama_base_url = st.text_input('Ollama base URL', value=current.ollama_base_url)
        local_synthesis_model = st.text_input('Local synthesis model', value=current.local_synthesis_model)
        local_vlm_model = st.text_input('Local VLM model', value=current.local_vlm_model)
        enable_vlms = st.checkbox('Authorize VLMs', value=current.enable_vlms)
        raw_power_weight = st.slider('Raw Power (quality)', 0.0, 1.0, float(current.raw_power_weight), 0.05)
        frugality_weight = st.slider('Frugality (energy saving)', 0.0, 1.0, float(current.frugality_weight), 0.05)
        independence_local_only = st.checkbox('Independence (local only)', value=current.independence_local_only)
        paid_openrouter_models = st.text_area('Paid models via OpenRouter (one per line)', value='\n'.join(current.paid_openrouter_models))
        selected_ollama_models = st.text_area('Preferred Ollama models (one per line)', value='\n'.join(current.selected_ollama_models))
        submitted = st.form_submit_button('Save settings')
    if submitted:
        payload = AppSettingsPayload(
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            gemini_api_key=gemini_api_key,
            perplexity_api_key=perplexity_api_key,
            openai_compatible_api_key=openai_compatible_api_key,
            openai_compatible_base_url=openai_compatible_base_url,
            openai_compatible_model=openai_compatible_model,
            ollama_base_url=ollama_base_url,
            local_synthesis_model=local_synthesis_model,
            local_vlm_model=local_vlm_model,
            enable_vlms=enable_vlms,
            raw_power_weight=raw_power_weight,
            frugality_weight=frugality_weight,
            independence_local_only=independence_local_only,
            paid_openrouter_models=[line.strip() for line in paid_openrouter_models.splitlines() if line.strip()],
            selected_ollama_models=[line.strip() for line in selected_ollama_models.splitlines() if line.strip()],
        )
        api_post('/api/settings', payload.model_dump())
        st.success('Settings saved locally.')


def render_monitoring() -> None:
    """Render the usage monitoring dashboard."""
    st.header('Monitoring')
    telemetry = api_get('/api/telemetry')
    if not telemetry:
        st.info('No telemetry yet. Send a prompt first.')
        return
    rows: List[Dict[str, Any]] = []
    for record in telemetry:
        for response in record['model_responses']:
            rows.append(
                {
                    'created_at': record['created_at'],
                    'conversation_id': record['conversation_id'],
                    'model_id': response['model_id'],
                    'provider': response['provider'],
                    'latency_s': response['latency_s'],
                    'energy_kwh': response['energy_kwh'],
                    'carbon_g': response['carbon_g'],
                    'cost_usd': response['cost_usd'],
                    'top_capability': max(record['router_decision']['categories'], key=record['router_decision']['categories'].get),
                }
            )
    frame = pd.DataFrame(rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Calls', len(frame))
    c2.metric('Energy (kWh)', f"{frame['energy_kwh'].sum():.4f}")
    c3.metric('CO₂e (g)', f"{frame['carbon_g'].sum():.1f}")
    c4.metric('Cost (USD)', f"{frame['cost_usd'].sum():.4f}")
    st.dataframe(frame, use_container_width=True)
    st.caption('Energy and carbon are estimated using a lightweight Green-Algorithms-inspired formula based on runtime, power draw, memory, PUE, and carbon intensity.')


def render_prompting() -> None:
    """Render the simplest possible prompt UI: left history, right prompt area."""
    st.header('Prompt')
    conversations = api_get('/api/conversations')
    left, right = st.columns([1, 3])
    with left:
        st.subheader('History')
        if not conversations:
            st.write('No flights yet.')
        conversation_options = {conversation['title']: conversation['conversation_id'] for conversation in conversations}
        selected_title = st.radio('Conversations', list(conversation_options.keys()) or ['No conversation'], label_visibility='collapsed')
        selected_conversation_id = conversation_options.get(selected_title)
        if st.button('New conversation'):
            selected_conversation_id = None
            st.session_state['selected_conversation_id'] = None
        if selected_conversation_id:
            st.session_state['selected_conversation_id'] = selected_conversation_id
    with right:
        st.subheader('Prompting')
        runtime_settings = AppSettingsPayload.model_validate(api_get('/api/settings'))
        prompt = st.text_area('Ask Roitelet', height=200, placeholder='Describe your question or task...')
        if st.button('Send', type='primary') and prompt.strip():
            payload = {
                'prompt': prompt,
                'conversation_id': st.session_state.get('selected_conversation_id'),
                'top_k': 3,
                'shadow_full_pool': True,
                'preferences': {
                    'raw_power': runtime_settings.raw_power_weight,
                    'frugality': runtime_settings.frugality_weight,
                    'independence': runtime_settings.independence_local_only,
                    'allow_vlms': runtime_settings.enable_vlms,
                },
            }
            result = api_post('/api/chat', payload)
            st.session_state['selected_conversation_id'] = result['conversation_id']
            st.success('Flight complete.')
            st.markdown(result['synthesis']['content'])
            with st.expander('Router'):
                st.json(result['router'])
            with st.expander('Model responses'):
                st.json(result['responses'])
        selected_conversation_id = st.session_state.get('selected_conversation_id')
        if selected_conversation_id:
            conversation = api_get(f'/api/conversations/{selected_conversation_id}')
            st.divider()
            for message in conversation['messages']:
                if message['role'] == 'user':
                    st.markdown(f"**You**\n\n{message['content']}")
                else:
                    st.markdown(f"**Roitelet**\n\n{message['content']}")


def main() -> None:
    """Run the Streamlit control room."""
    page_config()
    require_login()
    page = st.sidebar.radio('Section', ['Prompt', 'Configuration', 'Monitoring'])
    if page == 'Prompt':
        render_prompting()
    elif page == 'Configuration':
        render_settings()
    else:
        render_monitoring()


if __name__ == '__main__':
    main()
