"""Optional policy-only template setting; an unset variable preserves native behavior.

Keep identical copies beside each independent trainer and evaluation environment.
This setting never changes the training Judge or the method's user instructions.
"""
import os


def policy_chat_template_kwargs(existing=None):
    options = dict(existing or {})
    value = os.environ.get('POLICY_ENABLE_THINKING')
    if value is None:
        return options
    if value not in ('true', 'false'):
        raise ValueError('POLICY_ENABLE_THINKING must be true or false when set')
    enabled = value == 'true'
    if 'enable_thinking' in options and options['enable_thinking'] is not enabled:
        raise ValueError('Conflicting policy enable_thinking template settings')
    options['enable_thinking'] = enabled
    return options
