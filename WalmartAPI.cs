using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class WalmartAPI : MonoBehaviour
{
    IEnumerator Start()
    {
        string json = "{\"question\": \"Show me laptops under 50000\", \"store\": \"New York\"}";
        UnityWebRequest req = new UnityWebRequest("https://your-render-url.onrender.com/ask", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        req.uploadHandler = new UploadHandlerRaw(bodyRaw);
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.Log("Error: " + req.error);
        }
        else
        {
            Debug.Log("Response: " + req.downloadHandler.text);
        }
    }
}
