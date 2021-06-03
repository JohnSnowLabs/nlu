import streamlit as st
def _set_block_container_style(
        max_width: int = 1200,
        max_width_100_percent: bool = True,
        set_colors : bool = False ,
        COLOR : str = 'blue',
        BACKGROUND_COLOR : str = 'white',
):
    # .reportview-container .main .block-container
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    if set_colors:
        st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            {max_width_str}

        }}
        
     .element-container div:nth-child(1) {{
              margin: auto;

        }}
        
        .reportview-container .main {{
            color: {COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
        
    </style>
    """,
            unsafe_allow_html=True,
        )
    else :
        st.markdown(
            f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
          margin: auto;

    }}
     .element-container div:nth-child(1) {{
              margin: auto;

        }}
        
    textarea {{
        background-color:green;
    }}
</style>
""",
            unsafe_allow_html=True,
        )
