import streamlit as st
import anthropic
import os
import json
import time
from PIL import Image
import io
import base64
from datetime import datetime
import csv
# Avoid importing pandas

# Set page configuration
st.set_page_config(
    page_title="Transkriptionsassistent för Manuskript",
    page_icon="📜",
    layout="wide"
)

# Define a local wrapper function for Anthropic client creation
def create_anthropic_client(api_key):
    """En enkel wrapper för att skapa en Anthropic-klient oavsett SDK-version."""
    import anthropic
    import inspect
    
    try:
        # Kontrollera om Client-klassen finns (äldre versioner)
        if hasattr(anthropic, 'Client'):
            # För äldre versioner av SDK
            return anthropic.Client(api_key=api_key)
        # För nyare versioner av SDK
        elif hasattr(anthropic, 'Anthropic'):
            return anthropic.Anthropic(api_key=api_key)
        else:
            raise AttributeError("Kunde inte hitta varken Client eller Anthropic i SDK")
    except TypeError as e:
        # Om vi får TypeError, prova att skapa klienten på annat sätt
        st.sidebar.warning(f"Kompatibilitetsproblem: {str(e)}")
        
        # För vissa versioner kan proxies-parametern orsaka problem
        # Skapa ett nytt objekt med bara api_key
        if hasattr(anthropic, 'Client'):
            # Skapa ett objekt av Client-klassen
            client = object.__new__(anthropic.Client)
            # Sätt api_key direkt
            client.api_key = api_key
            return client
        elif hasattr(anthropic, 'Anthropic'):
            # Skapa ett objekt av Anthropic-klassen
            client = object.__new__(anthropic.Anthropic)
            # Sätt api_key direkt
            client.api_key = api_key
            return client
        else:
            raise ValueError("Kunde inte skapa klient med någon metod")

# Initialize Anthropic client using the wrapper
@st.cache_resource
def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        raise ValueError("Ingen Anthropic API-nyckel hittades. Ange ANTHROPIC_API_KEY som miljövariabel.")
    
    return create_anthropic_client(api_key)

# Convert image to base64 for Anthropic API
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_workflow_stage" not in st.session_state:
    st.session_state.current_workflow_stage = "upload"
if "current_iteration" not in st.session_state:
    st.session_state.current_iteration = 0
if "default_prompt" not in st.session_state:
    st.session_state.default_prompt = "Vänligen transkribera den handskrivna texten i denna manuskriptbild så noggrant som möjligt. Inkludera endast den transkriberade texten utan någon ytterligare kommentar."
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "training"  # "training" or "direct"
if "direct_mode_type" not in st.session_state:
    st.session_state.direct_mode_type = "Enstaka sida"  # "Enstaka sida" or "Bulk-transkription (flera sidor)"
if "training_metadata" not in st.session_state:
    st.session_state.training_metadata = {
        "name": "Onamngiven träningssession",
        "description": "Ingen beskrivning",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": 0
    }

# Function to save training history to JSON
def save_training_history():
    data = {
        "conversation_history": st.session_state.conversation_history,
        "metadata": st.session_state.training_metadata
    }
    return json.dumps(data, ensure_ascii=False)

# Function to load training history from JSON
def load_training_history(json_string):
    try:
        data = json.loads(json_string)
        st.session_state.conversation_history = data["conversation_history"]
        st.session_state.training_metadata = data["metadata"]
        return True
    except Exception as e:
        st.error(f"Fel vid inläsning av träningshistorik: {str(e)}")
        return False

# Function to handle the transcription process - Compatible with multiple SDK versions
def process_transcription(image, prompt, update_history=True):
    client = get_client()
    base64_image = image_to_base64(image)
    
    # Check if we're using an older version of the SDK without vision support
    import anthropic
    import pkg_resources
    
    # Try to get the SDK version
    try:
        sdk_version = pkg_resources.get_distribution("anthropic").version
        use_old_version = pkg_resources.parse_version(sdk_version) < pkg_resources.parse_version("0.17.0")
    except:
        # If we can't determine the version, assume it's too old
        use_old_version = True
    
    # For older versions that don't support vision
    if use_old_version:
        st.warning("Din version av Anthropic SDK stödjer inte bildanalys. Använder enbart textprompt.")
        
        # Create a simple prompt asking to transcribe handwritten text
        simple_prompt = f"""
        Du behöver transkribera en bild av handskriven text som jag inte kan visa dig. 
        Men jag ber dig svara på detta meddelande som om du hade sett bilden och transkriberat texten.
        
        Jag ber dig skriva:
        "Jag har analyserat den handskrivna texten och här är transkriptionen:
        
        [Transkription saknas - tekniskt problem]"
        
        Skriv bara exakt så, inget mer, eftersom detta är ett tekniskt test.
        """
        
        # Try to use the completion API if available
        if hasattr(client, 'completion'):
            try:
                # Legacy prompt format
                if hasattr(anthropic, 'HUMAN_PROMPT'):
                    # If SDK has these constants
                    human_prompt = anthropic.HUMAN_PROMPT
                    ai_prompt = anthropic.AI_PROMPT
                else:
                    # Hardcoded values if constants aren't available
                    human_prompt = "\n\nHuman: "
                    ai_prompt = "\n\nAssistant: "
                
                legacy_prompt = human_prompt + simple_prompt + ai_prompt
                
                response = client.completion(
                    prompt=legacy_prompt,
                    model="claude-2",  # Older model compatible with older SDK
                    max_tokens_to_sample=1000,
                    stop_sequences=[human_prompt]
                )
                
                if hasattr(response, 'completion'):
                    transcription = response.completion
                else:
                    transcription = str(response)
                
                # Return with notice about vision limitation
                return "[BEGRÄNSNING: Bildtranskription stöds inte med denna version av Anthropic SDK]\n\n" + transcription
            except Exception as e:
                st.error(f"Fel vid API-anrop: {str(e)}")
                return "Ett tekniskt fel uppstod. Din version av Anthropic SDK stödjer inte bildanalys."
    
    # For newer SDK versions, proceed with vision support
    # Construct the complete message history for context
    messages = []
    
    # Add all previous conversation history
    for msg in st.session_state.conversation_history:
        messages.append(msg)
    
    # Create the user message with the current image and prompt
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
    
    # Add to the messages for the API call
    messages.append(user_message)
    
    try:
        # Check if we're using the modern messages API
        if hasattr(client, 'messages') and hasattr(client.messages, 'create'):
            # Modern API (0.13.0+)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages
            )
            transcription = response.content[0].text
        else:
            raise ValueError("Denna version av Anthropic SDK stödjer inte Claude Vision.")
    
    except Exception as e:
        st.error(f"Fel vid API-anrop: {str(e)}")
        return f"Ett fel uppstod: {str(e)}. För att använda bildtranskribering, vänligen använd en modernare version av Anthropic SDK som stödjer Claude Vision."
    
    # If we should update the history (training mode), add the exchange to conversation history
    if update_history:
        st.session_state.conversation_history.append(user_message)
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": transcription
        })
    
    return transcription

# Main app title
st.title("Transkriptionsassistent för Manuskript")
st.write("Ladda upp bilder av handskrivna manuskript och träna Claude att transkribera dem korrekt.")

# App mode selector
st.sidebar.header("Appläge")
app_mode = st.sidebar.radio(
    "Välj läge:",
    ["Träningsläge", "Direktläge"],
    index=0 if st.session_state.app_mode == "training" else 1
)

# Update app mode in session state
st.session_state.app_mode = "training" if app_mode == "Träningsläge" else "direct"

# Sidebar for app controls and settings
with st.sidebar:
    st.header("Inställningar")
    st.write("Modell: claude-3-5-sonnet-20241022")
    
    # Training history management
    st.divider()
    
    # In direct mode, allow loading training history
    if st.session_state.app_mode == "direct":
        st.header("Ladda träningshistorik")
        uploaded_history = st.file_uploader("Välj en sparad träningsfil (.json)", type=["json"])
        
        if uploaded_history is not None:
            if st.button("Ladda träningshistorik"):
                content = uploaded_history.read().decode("utf-8")
                success = load_training_history(content)
                if success:
                    st.success(f"Träningshistorik laddad: {st.session_state.training_metadata['name']} ({st.session_state.training_metadata['iterations']} iterationer)")
                    # Reset workflow stage but keep conversation history
                    st.session_state.current_workflow_stage = "upload"
                    if "current_image" in st.session_state:
                        del st.session_state.current_image
                    if "current_transcription" in st.session_state:
                        del st.session_state.current_transcription
                    st.rerun()
    
    # In training mode, allow saving training history
    elif st.session_state.app_mode == "training" and len(st.session_state.conversation_history) > 0:
        st.header("Spara träningshistorik")
        
        # Edit metadata for the training session
        st.session_state.training_metadata["name"] = st.text_input(
            "Namn på träningssessionen:", 
            value=st.session_state.training_metadata["name"]
        )
        
        st.session_state.training_metadata["description"] = st.text_area(
            "Beskrivning:", 
            value=st.session_state.training_metadata["description"],
            height=100
        )
        
        # Update metadata
        st.session_state.training_metadata["iterations"] = st.session_state.current_iteration
        st.session_state.training_metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if st.button("Spara träningshistorik"):
            json_data = save_training_history()
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manuscript_training_{now}.json"
            
            st.download_button(
                label="Ladda ner träningsfil",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
    
    st.divider()
    
    # Show current status
    if st.session_state.app_mode == "training":
        st.header("Träningsstatistik")
        st.write(f"**Aktuell iteration:** {st.session_state.current_iteration + 1}")
        st.write(f"**Antal meddelanden i historik:** {len(st.session_state.conversation_history)}")
    else:
        st.header("Direktlägesstatus")
        if len(st.session_state.conversation_history) > 0:
            st.write(f"**Aktiv träningsprofil:** {st.session_state.training_metadata['name']}")
            st.write(f"**Träningsiterationer:** {st.session_state.training_metadata['iterations']}")
            st.info("I direktläge bevaras den ursprungliga träningshistoriken och nya transkriptioner läggas inte till i kontexten.")
        else:
            st.warning("Ingen träningshistorik laddad. Ladda en träningsprofil för bästa resultat.")
    
    # Reset button
    if st.button("Återställ applikationen"):
        st.session_state.conversation_history = []
        st.session_state.current_workflow_stage = "upload"
        st.session_state.current_iteration = 0
        st.session_state.training_metadata = {
            "name": "Onamngiven träningssession",
            "description": "Ingen beskrivning",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": 0
        }
        if "current_image" in st.session_state:
            del st.session_state.current_image
        if "current_transcription" in st.session_state:
            del st.session_state.current_transcription
        if "current_reflection" in st.session_state:
            del st.session_state.current_reflection
        st.rerun()

# Main content area - first check which mode we're in
if st.session_state.app_mode == "training":
    # TRAINING MODE
    st.info("**Träningsläge:** I detta läge tränar du Claude iterativt att transkribera manuskript genom att ge feedback på transkriptionerna.")
    
    # Create two columns for layout in training mode
    col1, col2 = st.columns([1, 1])
    
    # STAGE 1: UPLOAD IMAGE
    if st.session_state.current_workflow_stage == "upload":
        with col1:
            st.subheader("Steg 1: Ladda upp ett manuskript")
            uploaded_file = st.file_uploader("Välj en bild av ett handskrivet manuskript", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Process the uploaded image
                image = Image.open(uploaded_file)
                st.session_state.current_image = image
                st.image(image, caption="Uppladdat manuskript", use_column_width=True)
                
                # Move to the next stage
                st.session_state.current_workflow_stage = "prompt"
                st.rerun()

    # STAGE 2: CONFIGURE PROMPT AND TRANSCRIBE
    elif st.session_state.current_workflow_stage == "prompt" and "current_image" in st.session_state:
        with col1:
            st.subheader("Uppladdat manuskript")
            st.image(st.session_state.current_image, caption=f"Iteration {st.session_state.current_iteration + 1}", use_column_width=True)
        
        with col2:
            st.subheader("Steg 2: Anpassa prompt och transkribera")
            
            # Show the prompt input field
            custom_prompt = st.text_area(
                "Prompt för Claude:", 
                value=st.session_state.default_prompt,
                height=150
            )
            
            # Button to start transcription
            if st.button("Starta transkription"):
                with st.spinner("Claude transkriberar manuskriptet..."):
                    try:
                        # Save the custom prompt for future use
                        st.session_state.default_prompt = custom_prompt
                        
                        # Get transcription from Claude
                        transcription = process_transcription(st.session_state.current_image, custom_prompt)
                        
                        # Save the transcription and update the conversation history
                        st.session_state.current_transcription = transcription
                        
                        # process_transcription now handles adding to conversation history
                        
                        # Move to the next stage
                        st.session_state.current_workflow_stage = "transcribe"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ett fel uppstod vid transkriberingen: {str(e)}")

    # STAGE 3: SHOW TRANSCRIPTION AND GET FEEDBACK
    elif st.session_state.current_workflow_stage == "transcribe" and "current_transcription" in st.session_state:
        with col1:
            st.subheader("Uppladdat manuskript")
            st.image(st.session_state.current_image, caption=f"Iteration {st.session_state.current_iteration + 1}", use_column_width=True)
        
        with col2:
            st.subheader("Steg 3: Claudes transkription")
            st.text_area("Claudes transkription:", value=st.session_state.current_transcription, height=200, disabled=True)
            
            st.subheader("Steg 4: Din feedback")
            st.write("Ange den korrekta transkriptionen av manuskriptet:")
            
            correct_transcription = st.text_area("Korrekt transkription:", height=200)
            
            if st.button("Skicka feedback"):
                with st.spinner("Claude reflekterar över feedbacken..."):
                    try:
                        # Create the feedback prompt
                        feedback_prompt = f"""Här är den korrekta transkriptionen av manuskriptet:

{correct_transcription}

Jämför din transkription med den korrekta versionen ovan. 
1. Vilka specifika fel gjorde du?
2. Vilka aspekter av handskriften var svåra att tyda?
3. Vad kan du lära dig för att förbättra framtida transkriptioner?

Var specifik i din analys för att kunna förbättra din förmåga att transkribera liknande manuskript i framtiden."""
                        
                        # Get Claude's reflection
                        reflection = process_transcription(st.session_state.current_image, feedback_prompt)
                        
                        # Store the feedback exchange in conversation history
                        # Note: We always update history for feedback in training mode
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": feedback_prompt
                        })
                        
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": reflection
                        })
                        
                        # Increment the iteration counter
                        st.session_state.current_iteration += 1
                        
                        # Display the reflection
                        st.session_state.current_reflection = reflection
                        st.session_state.current_workflow_stage = "reflection"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ett fel uppstod vid bearbetning av feedback: {str(e)}")

    # STAGE 4: SHOW REFLECTION AND PREPARE FOR NEXT ITERATION
    elif st.session_state.current_workflow_stage == "reflection" and "current_reflection" in st.session_state:
        with col1:
            st.subheader("Uppladdat manuskript")
            st.image(st.session_state.current_image, caption=f"Iteration {st.session_state.current_iteration}", use_column_width=True)
        
        with col2:
            st.subheader("Claudes reflektion")
            st.write(st.session_state.current_reflection)
            
            st.divider()
            
            st.write("Claude har nu lärt sig från denna iteration och är redo för nästa manuskript.")
            if st.button("Fortsätt med nästa manuskript"):
                # Reset for next iteration but keep the conversation history
                if "current_image" in st.session_state:
                    del st.session_state.current_image
                if "current_transcription" in st.session_state:
                    del st.session_state.current_transcription
                if "current_reflection" in st.session_state:
                    del st.session_state.current_reflection
                
                st.session_state.current_workflow_stage = "upload"
                st.rerun()

else:
    # DIRECT MODE - Start with the mode selection at the very top
    st.info("**Direktläge:** I detta läge kommer Claude att direkt transkribera manuskript baserat på tidigare träning utan att kräva feedback.")
    
    # Place the transcription method selection at the very top of the direct mode
    direct_mode_type = st.radio(
        "Välj transkriptionsmetod:",
        ["Enstaka sida", "Bulk-transkription (flera sidor)"],
        index=0 if st.session_state.direct_mode_type == "Enstaka sida" else 1
    )
    
    # Store the selected mode in session state
    st.session_state.direct_mode_type = direct_mode_type
    
    # Create two columns for layout in direct mode
    col1, col2 = st.columns([1, 1])
    
    # Based on the selected transcription method, show the appropriate interface
    if direct_mode_type == "Enstaka sida":
        # Single page transcription mode
        with col1:
            st.subheader("Ladda upp ett manuskript")
            uploaded_file = st.file_uploader("Välj en bild av ett handskrivet manuskript", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Process the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Manuskript för direkt transkription", use_column_width=True)
                
                # Store the image for processing
                st.session_state.direct_mode_image = image
        
        with col2:
            # Show prompt editor and transcribe button if an image is uploaded
            if "direct_mode_image" in st.session_state:
                st.subheader("Anpassa prompt och transkribera")
                
                # Show the prompt input field
                direct_prompt = st.text_area(
                    "Prompt för Claude:", 
                    value=st.session_state.default_prompt,
                    height=150
                )
                
                # Button to start direct transcription
                if st.button("Starta direkt transkription"):
                    with st.spinner("Claude transkriberar manuskriptet..."):
                        try:
                            # Get transcription from Claude using all previous training
                            # Pass update_history=False to prevent adding to conversation history in direct mode
                            transcription = process_transcription(
                                st.session_state.direct_mode_image, 
                                direct_prompt,
                                update_history=False
                            )
                            
                            # Display the result
                            st.session_state.direct_transcription = transcription
                            
                        except Exception as e:
                            st.error(f"Ett fel uppstod vid transkriberingen: {str(e)}")
                
                # Show the direct transcription result if available
                if "direct_transcription" in st.session_state:
                    st.subheader("Transkriptionsresultat")
                    st.text_area(
                        "Claudes transkription:", 
                        value=st.session_state.direct_transcription, 
                        height=300
                    )
                    
                    # Option to copy to clipboard
                    if st.button("Kopiera till urklipp"):
                        st.code(st.session_state.direct_transcription)
                        st.success("Transkription kopierad till urklipp!")
                    
                    # Button to clear for next transcription
                    if st.button("Rensa och transkribera en ny bild"):
                        if "direct_transcription" in st.session_state:
                            del st.session_state.direct_transcription
                        if "direct_mode_image" in st.session_state:
                            del st.session_state.direct_mode_image
                        st.rerun()
            else:
                st.info("Ladda upp en bild för att börja transkribera.")
    
    else:  # Bulk transcription mode
        st.subheader("Bulk-transkription av flera manuskript")
        
        # Initialize session state for bulk transcription if not already done
        if "bulk_transcription_results" not in st.session_state:
            st.session_state.bulk_transcription_results = []
        
        if "bulk_transcription_completed" not in st.session_state:
            st.session_state.bulk_transcription_completed = False
        
        if "bulk_transcription_progress" not in st.session_state:
            st.session_state.bulk_transcription_progress = 0
        
        # Allow uploading multiple files
        uploaded_files = st.file_uploader(
            "Välj flera bilder av handskrivna manuskript", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        # Show the prompt input field
        bulk_prompt = st.text_area(
            "Prompt för Claude (används för alla bilder):", 
            value=st.session_state.default_prompt,
            height=150
        )
        
        # Button to start bulk transcription
        if uploaded_files and st.button("Starta bulk-transkription"):
            # Reset results
            st.session_state.bulk_transcription_results = []
            st.session_state.bulk_transcription_completed = False
            st.session_state.bulk_transcription_progress = 0
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress_percent = int(100 * i / len(uploaded_files))
                progress_bar.progress(progress_percent)
                status_text.text(f"Transkriberar fil {i+1} av {len(uploaded_files)}: {uploaded_file.name}")
                
                try:
                    # Process the image
                    image = Image.open(uploaded_file)
                    
                    # Get transcription from Claude
                    transcription = process_transcription(
                        image, 
                        bulk_prompt,
                        update_history=False
                    )
                    
                    # Store the result
                    st.session_state.bulk_transcription_results.append({
                        "filename": uploaded_file.name,
                        "transcription": transcription
                    })
                    
                except Exception as e:
                    # Store the error
                    st.session_state.bulk_transcription_results.append({
                        "filename": uploaded_file.name,
                        "transcription": f"FEL: {str(e)}"
                    })
            
            # Complete the progress bar
            progress_bar.progress(100)
            status_text.text(f"Transkribering klar! {len(uploaded_files)} filer bearbetade.")
            
            # Mark as completed
            st.session_state.bulk_transcription_completed = True
            st.rerun()
        
        # If bulk transcription has been completed, show results and download option
        if st.session_state.bulk_transcription_completed and st.session_state.bulk_transcription_results:
            st.subheader("Bulk-transkriptionsresultat")
            
            # Create a simple table display without pandas
            st.write("### Transkriptionsresultat")
            
            # Display the results in a table format
            table_data = [["Filnamn", "Transkription"]]
            for result in st.session_state.bulk_transcription_results:
                table_data.append([
                    result.get('filename', 'Okänd fil'),
                    result.get('transcription', 'Ingen transkription')
                ])
            
            # Use Streamlit's native table display
            st.table(table_data)
            
            # Generate CSV data manually without pandas
            csv_content = io.StringIO()
            csv_writer = csv.writer(csv_content)
            csv_writer.writerow(["filename", "transcription"])
            for result in st.session_state.bulk_transcription_results:
                csv_writer.writerow([
                    result.get('filename', ''),
                    result.get('transcription', '')
                ])
            
            # Create a download button for CSV
            st.download_button(
                label="Ladda ner resultat som CSV",
                data=csv_content.getvalue(),
                file_name="transkriptionsresultat.csv",
                mime="text/csv"
            )
            
            # Also offer JSON download as alternative
            json_data = json.dumps(st.session_state.bulk_transcription_results, ensure_ascii=False)
            st.download_button(
                label="Ladda ner resultat som JSON",
                data=json_data,
                file_name="transkriptionsresultat.json",
                mime="application/json"
            )
            
            # Option to clear results
            if st.button("Rensa resultat och transkribera nya filer"):
                st.session_state.bulk_transcription_results = []
                st.session_state.bulk_transcription_completed = False
                st.rerun()

# Show training history
with st.expander("Visa träningshistorik"):
    if len(st.session_state.conversation_history) > 0:
        iteration = 1
        message_index = 0
        
        while message_index < len(st.session_state.conversation_history):
            # Try to find a complete iteration (4 messages)
            if message_index + 3 < len(st.session_state.conversation_history):
                # Get the messages
                image_msg = st.session_state.conversation_history[message_index]
                transcription_msg = st.session_state.conversation_history[message_index + 1]
                
                # Check if this is a training iteration with feedback
                if message_index + 3 < len(st.session_state.conversation_history) and "Här är den korrekta transkriptionen" in st.session_state.conversation_history[message_index + 2]["content"]:
                    # This is a training iteration with feedback
                    feedback_msg = st.session_state.conversation_history[message_index + 2]
                    reflection_msg = st.session_state.conversation_history[message_index + 3]
                    
                    st.write(f"### Träningsiteration {iteration}")
                    
                    # Display transcription
                    st.write("**Claudes transkription:**")
                    st.write(transcription_msg["content"])
                    
                    # Display reflection
                    st.write("**Claudes reflektion:**")
                    st.write(reflection_msg["content"])
                    
                    st.divider()
                    
                    # Move to next iteration
                    iteration += 1
                    message_index += 4
                else:
                    # Skip to next message - we shouldn't have direct mode messages in history anymore
                    message_index += 1
            else:
                # Handle remaining messages
                st.write("*Ofullständig träningsiteration*")
                break
    else:
        st.write("Ingen träningshistorik än.")
