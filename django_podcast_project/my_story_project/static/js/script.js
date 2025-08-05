// static/js/script.js (PHIÊN BẢN HOÀN THIỆN CUỐI CÙNG)

document.addEventListener('DOMContentLoaded', () => {
    setupThemeToggle();
    setupCharacterPresets();
    setupStoryForm();
});

function setupStoryForm() {
    const generateButton = document.getElementById('generate-button');
    if (!generateButton) return;

    generateButton.addEventListener('click', function() {
        const plotline = document.getElementById('plotline-input').value;
        const char1_name = document.getElementById('char1-select').value;
        const char2_name = document.getElementById('char2-select').value;
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

        if (!plotline || !char1_name || !char2_name) {
            alert("Please fill in the plotline and select two characters.");
            return;
        }
        if (char1_name === char2_name) {
            alert("Please select two different characters.");
            return;
        }

        const formData = new FormData();
        formData.append('csrfmiddlewaretoken', csrfToken);
        formData.append('plotline', plotline);
        formData.append('character1', char1_name);
        formData.append('character2', char2_name);

        document.getElementById('initial-message').style.display = 'none';
        document.getElementById('story-content').innerHTML = '';
        document.getElementById('story-content').style.display = 'none';
        document.getElementById('loading-indicator').style.display = 'flex';
        document.getElementById('loading-message-text').textContent = 'Sending request to the AI...';
        generateButton.disabled = true;
        generateButton.innerText = 'Generating...';

        fetch(window.APP_CONFIG.CREATE_URL, {
                method: 'POST',
                body: formData,
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        throw new Error(`HTTP error! Status: ${response.status}, Body: ${text}`)
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.task_id) {
                    checkTaskStatus(data.task_id);
                } else {
                    handleError(data.error || "Server did not return a task ID.");
                }
            })
            .catch(error => {
                console.error('Error submitting form:', error);
                handleError("Failed to submit the story request. Check the Django server console for errors.");
            });
    });
}

function checkTaskStatus(taskId) {
    const statusUrl = `${window.APP_CONFIG.STATUS_URL_BASE}${taskId}/`;
    const intervalId = setInterval(() => {
        fetch(statusUrl)
            .then(response => response.ok ? response.json() : {
                state: 'FAILURE',
                details: {
                    status: `Server error: ${response.status}`
                }
            })
            .then(data => {
                console.log('Checking status...', data);
                const loadingMessageEl = document.getElementById('loading-message-text');
                if (data.details && data.details.status) {
                    loadingMessageEl.textContent = data.details.status;
                }
                if (data.state === 'SUCCESS') {
                    clearInterval(intervalId);
                    displayStory(data.result);
                } else if (data.state === 'FAILURE') {
                    clearInterval(intervalId);
                    handleError(data.details ? data.details.status : "The AI task failed.");
                }
            })
            .catch(error => {
                clearInterval(intervalId);
                console.error('Error checking task status:', error);
                handleError("Could not check the story status. The API endpoint might be wrong.");
            });
    }, 4000);
}

function displayStory(storyData) {
    const loadingIndicatorDiv = document.getElementById('loading-indicator');
    const storyContentDiv = document.getElementById('story-content');
    const generateButton = document.getElementById('generate-button');
    loadingIndicatorDiv.style.display = 'none';

    if (storyData && storyData.characters && storyData.chapters) {
        storyContentDiv.innerHTML = `
            <div id="book-container">
                <div id="full-podcast-player" style="display: none;">
                    <h2>Listen to the Full Story</h2>
                    <audio id="podcast-audio-element" controls></audio>
                </div>
                <div id="story-introduction">
                    <div id="character-portraits"></div>
                    <p id="introduction-text"></p>
                </div>
                <div id="chapters-container"></div>
            </div>`;

        if (storyData.full_podcast_url) {
            document.getElementById('podcast-audio-element').src = storyData.full_podcast_url;
            document.getElementById('full-podcast-player').style.display = 'block';
        }

        const charPortraitsContainer = document.getElementById('character-portraits');
        let portraitsHtml = '';
        storyData.characters.forEach(char => {
            const originalCharData = window.CHARACTERS_DATA.find(c => c.name === char.name);
            const finalImageUrl = originalCharData ? originalCharData.static_image_url : '';
            portraitsHtml += `<div class="portrait-card">
                                  <img src="${finalImageUrl}" alt="Portrait of ${char.name}">
                                  <h3>${char.name}</h3>
                              </div>`;
        });
        charPortraitsContainer.innerHTML = portraitsHtml;

        if (storyData.introduction) {
            document.getElementById('introduction-text').textContent = storyData.introduction;
        }

        const chaptersContainer = document.getElementById('chapters-container');
        let chaptersHtml = '';
        storyData.chapters.forEach(chapter => {
            const imageUrl = chapter.image_url ? `<img src="${chapter.image_url}" alt="Illustration for ${chapter.title}">` : '';

            let cleanContent = chapter.content;
            if (cleanContent.trim().toLowerCase().startsWith(chapter.title.toLowerCase())) {
                cleanContent = cleanContent.trim().slice(chapter.title.length).trim();
            }

            // === SỬA LỖI TYPOGRAPHY ===
            // 1. Tách văn bản thành các đoạn dựa trên dấu xuống dòng.
            // 2. Lọc bỏ các đoạn trống.
            // 3. Bọc mỗi đoạn còn lại trong cặp thẻ <p> của riêng nó.
            // 4. Nối tất cả lại.
            const paragraphs = cleanContent.split('\n')
                                           .filter(p => p.trim() !== '')
                                           .map(p => `<p>${p.trim()}</p>`)
                                           .join('');

            chaptersHtml += `
                <div class="chapter-container">
                    <div class="chapter-media">
                        ${imageUrl}
                    </div>
                    <div class="chapter-text">
                        <h2>
                            <span class="chapter-number">Chapter ${chapter.chapter}: </span>
                            <span class="chapter-title-text">${chapter.title}</span>
                        </h2>
                        ${paragraphs}
                    </div>
                </div>`;
        });
        chaptersContainer.innerHTML = chaptersHtml;
        storyContentDiv.style.display = 'block';

    } else {
        handleError("Received an unexpected data structure from the server.");
    }

    generateButton.disabled = false;
    generateButton.innerText = '✨ Generate Podcast ✨';
}

function handleError(message) {
    const storyContentDiv = document.getElementById('story-content');
    document.getElementById('loading-indicator').style.display = 'none';
    storyContentDiv.innerHTML = `<h2 style="color: #dc3545;">Oh no! Something went wrong.</h2><p>${message}</p>`;
    storyContentDiv.style.display = 'block';
    document.getElementById('generate-button').disabled = false;
    document.getElementById('generate-button').innerText = '✨ Generate Podcast ✨';
}

function setupThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;
    const body = document.body;

    function applyTheme() {
        if (localStorage.getItem('theme') === 'dark') {
            body.classList.add('dark-mode');
        } else {
            body.classList.remove('dark-mode');
        }
    }
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-mode');
        localStorage.setItem('theme', body.classList.contains('dark-mode') ? 'dark' : 'light');
    });
    applyTheme();
}

function setupCharacterPresets() {
    if (!window.CHARACTERS_DATA) {
        return;
    }

    function handleSelectChange(selectElement, bioTextareaElement) {
        const selectedName = selectElement.value;
        const selectedChar = window.CHARACTERS_DATA.find(char => char.name === selectedName);
        bioTextareaElement.value = selectedChar ? selectedChar.bio : '';
    }
    const char1Select = document.getElementById('char1-select');
    const char1Bio = document.getElementById('char1-bio');
    if (char1Select && char1Bio) {
        char1Select.addEventListener('change', () => handleSelectChange(char1Select, char1Bio));
        handleSelectChange(char1Select, char1Bio);
    }
    const char2Select = document.getElementById('char2-select');
    const char2Bio = document.getElementById('char2-bio');
    if (char2Select && char2Bio) {
        char2Select.addEventListener('change', () => handleSelectChange(char2Select, char2Bio));
        handleSelectChange(char2Select, char2Bio);
    }
}