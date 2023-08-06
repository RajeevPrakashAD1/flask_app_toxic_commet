console.log('connected');
document.getElementById('myform').addEventListener('submit', function(event) {
	event.preventDefault();
});

function submit2() {
	console.log('submitted');

	// Get the input text from the textarea element
	const inputText = document.getElementById('input').value;

	// Create a JSON object to send in the request body
	const jsonData = { input: inputText };
	console.log('jsonData', jsonData);
	fetch('http://127.0.0.1:5000/predict2', {
		body: JSON.stringify(jsonData),
		headers: {
			'Content-Type': 'application/json'
		},
		method: 'post'
	})
		.then(function(response) {
			if (response.ok) {
				return response.json();
			} else {
				console.log(response);
				return Promise.reject(response);
			}
		})
		.then(function(data) {
			console.log('data coming', data);
			// This is the JSON from our response
			// console.log(data);
			document.getElementById('toxic').innerHTML = '';
			if (data['predicted_class'].length == 0) {
				document.getElementById('toxic').innerHTML = 'Not toxic';
			} else {
				for (var i of data['predicted_class']) {
					document.getElementById('toxic').innerHTML += i + ',';
				}
			}
		})
		.catch(function(err) {
			// There was an error
			console.warn('Something went wrong.', err);
		});
}
