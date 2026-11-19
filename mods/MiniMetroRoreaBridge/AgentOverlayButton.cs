using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace MiniMetroRoreaBridge
{
    /// <summary>
    /// Renders a compact AI-agent toggle pill in the upper-right HUD during active map play
    /// using a screen-space overlay <see cref="Canvas"/> and procedural anti-aliased textures.
    /// </summary>
    public sealed class AgentOverlayButton : IDisposable
    {
        private const int TexW = 280;
        private const int TexH = 76;

        private GameObject _canvasObj;
        private RectTransform _canvasRect;
        private RectTransform _buttonRect;
        private RawImage _buttonImage;
        private RawImage _pulseDotImage;
        private RectTransform _pulseDotRect;

        private Texture2D _texOff;
        private Texture2D _texOffHover;
        private Texture2D _texOn;
        private Texture2D _texOnHover;
        private Texture2D _texError;
        private Texture2D _dotTex;

        private bool _wasMouseDown;
        private bool _initialized;
        private bool _disposed;

        /// <summary>
        /// Updates visibility, hover/active visuals, and hit-testing. Returns <c>true</c> on click.
        /// </summary>
        public bool Update(bool visible, bool isRunning, bool hasError)
        {
            if (_disposed) return false;
            EnsureCreated();
            if (_canvasObj == null) return false;

            if (!visible)
            {
                if (_canvasObj.activeSelf) _canvasObj.SetActive(false);
                return false;
            }

            if (!_canvasObj.activeSelf) _canvasObj.SetActive(true);

            float scale = Mathf.Clamp(Screen.height / 1080f, 0.75f, 1.5f);
            float width = 132f * scale;
            float height = 36f * scale;
            float marginRight = 24f * scale;
            float offsetTop = 112f * scale;

            Rect cr = _canvasRect != null ? _canvasRect.rect : new Rect(0f, 0f, 0f, 0f);
            float halfW = cr.width > 200f ? cr.width * 0.5f : Screen.width * 0.36f;
            float halfH = cr.height > 200f ? cr.height * 0.5f : Screen.height * 0.36f;

            _buttonRect.sizeDelta = new Vector2(width, height);
            _buttonRect.anchoredPosition = new Vector2(halfW - marginRight, halfH - offsetTop);

            float dotSize = 8f * scale;
            _pulseDotRect.sizeDelta = new Vector2(dotSize, dotSize);
            _pulseDotRect.anchoredPosition = new Vector2(-13f * scale, 0f);

            float minX = Screen.width * 0.5f + (halfW - marginRight - width) * (Screen.width / (halfW * 2f));
            float maxX = Screen.width * 0.5f + (halfW - marginRight) * (Screen.width / (halfW * 2f));
            float minY = Screen.height * 0.5f + (halfH - offsetTop - height) * (Screen.height / (halfH * 2f));
            float maxY = Screen.height * 0.5f + (halfH - offsetTop) * (Screen.height / (halfH * 2f));

            Vector3 mouse = Input.mousePosition;
            bool hovered = mouse.x >= minX && mouse.x <= maxX && mouse.y >= minY && mouse.y <= maxY;

            _buttonImage.texture = SelectTexture(isRunning, hasError, hovered);

            if (isRunning)
            {
                float pulse = 0.60f + 0.40f * Mathf.Sin(Time.realtimeSinceStartup * 4.5f);
                _pulseDotImage.color = new Color(0.22f, 0.98f, 0.84f, pulse);
            }
            else if (hasError)
            {
                _pulseDotImage.color = new Color(1.0f, 0.37f, 0.43f, 0.95f);
            }
            else
            {
                _pulseDotImage.color = new Color(0.45f, 0.53f, 0.61f, 0.85f);
            }

            bool mouseDown = Input.GetMouseButton(0);
            bool clicked = hovered && (Input.GetMouseButtonDown(0) || (mouseDown && !_wasMouseDown));
            _wasMouseDown = mouseDown;
            return clicked;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_canvasObj != null)
            {
                UnityEngine.Object.Destroy(_canvasObj);
                _canvasObj = null;
            }
            DestroyTex(ref _texOff);
            DestroyTex(ref _texOffHover);
            DestroyTex(ref _texOn);
            DestroyTex(ref _texOnHover);
            DestroyTex(ref _texError);
            DestroyTex(ref _dotTex);
            _initialized = false;
        }

        private Texture2D SelectTexture(bool isRunning, bool hasError, bool hovered)
        {
            if (isRunning) return hovered ? _texOnHover : _texOn;
            if (hasError) return _texError;
            return hovered ? _texOffHover : _texOff;
        }

        private void EnsureCreated()
        {
            if (_initialized) return;

            _texOff = CreateButtonTexture(
                fill: new Color32(22, 30, 42, 232),
                border: new Color32(74, 101, 125, 224),
                iconPrimary: new Color(0.66f, 0.78f, 0.89f, 1f),
                iconAccent: new Color(0.45f, 0.68f, 0.84f, 0.95f),
                textColor: new Color(0.86f, 0.90f, 0.95f, 0.96f),
                label: "AI · OFF");

            _texOffHover = CreateButtonTexture(
                fill: new Color32(32, 44, 60, 244),
                border: new Color32(114, 168, 206, 245),
                iconPrimary: new Color(0.78f, 0.88f, 0.97f, 1f),
                iconAccent: new Color(0.55f, 0.78f, 0.94f, 1f),
                textColor: new Color(0.95f, 0.97f, 1.00f, 1f),
                label: "AI · OFF");

            _texOn = CreateButtonTexture(
                fill: new Color32(11, 44, 54, 242),
                border: new Color32(44, 230, 200, 245),
                iconPrimary: new Color(0.24f, 0.98f, 0.85f, 1f),
                iconAccent: new Color(0.68f, 1.00f, 0.95f, 1f),
                textColor: new Color(0.90f, 1.00f, 0.97f, 1f),
                label: "AI · ON");

            _texOnHover = CreateButtonTexture(
                fill: new Color32(16, 58, 70, 248),
                border: new Color32(92, 250, 224, 255),
                iconPrimary: new Color(0.40f, 1.00f, 0.90f, 1f),
                iconAccent: new Color(0.80f, 1.00f, 0.97f, 1f),
                textColor: new Color(0.96f, 1.00f, 0.99f, 1f),
                label: "AI · ON");

            _texError = CreateButtonTexture(
                fill: new Color32(58, 20, 28, 242),
                border: new Color32(255, 94, 108, 245),
                iconPrimary: new Color(1.00f, 0.56f, 0.60f, 1f),
                iconAccent: new Color(1.00f, 0.76f, 0.78f, 1f),
                textColor: new Color(1.00f, 0.84f, 0.86f, 1f),
                label: "AI · ERR");

            _dotTex = CreateCircleTexture(24);

            _canvasObj = new GameObject("MiniMetroRoreaBridge_AgentHud");
            UnityEngine.Object.DontDestroyOnLoad(_canvasObj);

            Canvas canvas = _canvasObj.AddComponent<Canvas>();
            canvas.sortingOrder = 500;
            _canvasRect = _canvasObj.GetComponent<RectTransform>();

            var btnObj = new GameObject("AgentToggleButton");
            btnObj.transform.SetParent(_canvasObj.transform, worldPositionStays: false);
            _buttonRect = btnObj.AddComponent<RectTransform>();
            _buttonRect.anchorMin = new Vector2(0.5f, 0.5f);
            _buttonRect.anchorMax = new Vector2(0.5f, 0.5f);
            _buttonRect.pivot = new Vector2(1f, 1f);

            _buttonImage = btnObj.AddComponent<RawImage>();
            _buttonImage.raycastTarget = false;
            _buttonImage.texture = _texOff;

            var dotObj = new GameObject("AgentStatusDot");
            dotObj.transform.SetParent(btnObj.transform, worldPositionStays: false);
            _pulseDotRect = dotObj.AddComponent<RectTransform>();
            _pulseDotRect.anchorMin = new Vector2(1f, 0.5f);
            _pulseDotRect.anchorMax = new Vector2(1f, 0.5f);
            _pulseDotRect.pivot = new Vector2(1f, 0.5f);

            _pulseDotImage = dotObj.AddComponent<RawImage>();
            _pulseDotImage.raycastTarget = false;
            _pulseDotImage.texture = _dotTex;

            _canvasObj.SetActive(false);
            _initialized = true;
        }

        private static Texture2D CreateButtonTexture(
            Color32 fill,
            Color32 border,
            Color iconPrimary,
            Color iconAccent,
            Color textColor,
            string label)
        {
            var tex = new Texture2D(TexW, TexH, TextureFormat.RGBA32, mipChain: false)
            {
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };

            var pixels = new Color[TexW * TexH];
            float halfW = TexW * 0.5f - 2.0f;
            float halfH = TexH * 0.5f - 2.0f;
            float radius = halfH;
            float borderThickness = 3.2f;

            for (int y = 0; y < TexH; y++)
            {
                float py = (y + 0.5f) - TexH * 0.5f;
                for (int x = 0; x < TexW; x++)
                {
                    float px = (x + 0.5f) - TexW * 0.5f;
                    float dist = RoundedBoxSdf(px, py, halfW, halfH, radius);
                    float outerAlpha = Mathf.Clamp01(0.75f - dist);
                    if (outerAlpha <= 0f)
                    {
                        pixels[y * TexW + x] = Color.clear;
                        continue;
                    }

                    float innerT = Mathf.Clamp01(0.75f - (dist + borderThickness));
                    Color mixed = Color.Lerp(border, fill, innerT);
                    mixed.a *= outerAlpha;
                    pixels[y * TexW + x] = mixed;
                }
            }

            Vector2 starCenter = new Vector2(36f, TexH * 0.5f + 1.0f);
            float starRadius = 15.5f;
            Vector2 nodeA = starCenter + new Vector2(18f, 15f);
            Vector2 nodeB = starCenter + new Vector2(20f, -4f);
            Vector2 nodeC = starCenter + new Vector2(12f, -16f);

            for (int y = 10; y < TexH - 10; y++)
            {
                for (int x = 12; x < 68; x++)
                {
                    var p = new Vector2(x + 0.5f, y + 0.5f);
                    Vector2 d = p - starCenter;
                    float nx = Mathf.Abs(d.x) / starRadius;
                    float ny = Mathf.Abs(d.y) / starRadius;
                    float astroid = Mathf.Pow(nx + 1e-4f, 0.58f) + Mathf.Pow(ny + 1e-4f, 0.58f);
                    float starAlpha = Mathf.Clamp01((1.0f - astroid) * 5.0f + 0.5f);

                    float linkDist = Mathf.Min(
                        SegmentDist(p, starCenter + new Vector2(6f, 6f), nodeA),
                        Mathf.Min(
                            SegmentDist(p, starCenter + new Vector2(8f, -1.5f), nodeB),
                            SegmentDist(p, nodeB, nodeC)));
                    float linkAlpha = Mathf.Clamp01(1.8f - linkDist) * 0.85f;

                    float nodeDist = Mathf.Min(
                        Vector2.Distance(p, nodeA) - 3.6f,
                        Mathf.Min(Vector2.Distance(p, nodeB) - 3.2f, Vector2.Distance(p, nodeC) - 2.9f));
                    float nodeAlpha = Mathf.Clamp01(0.8f - nodeDist);

                    Color fg = Color.clear;
                    if (linkAlpha > 0f)
                        fg = new Color(iconAccent.r, iconAccent.g, iconAccent.b, iconAccent.a * linkAlpha);
                    if (nodeAlpha > 0f)
                        fg = AlphaOver(new Color(iconAccent.r, iconAccent.g, iconAccent.b, iconAccent.a * nodeAlpha), fg);
                    if (starAlpha > 0f)
                        fg = AlphaOver(new Color(iconPrimary.r, iconPrimary.g, iconPrimary.b, iconPrimary.a * starAlpha), fg);

                    if (fg.a > 0f)
                        pixels[y * TexW + x] = AlphaOver(fg, pixels[y * TexW + x]);
                }
            }

            DrawBitmapText(pixels, TexW, TexH, label, startX: 76, centerY: TexH / 2, scale: 3, textColor);

            var outPixels = new Color32[pixels.Length];
            for (int i = 0; i < pixels.Length; i++) outPixels[i] = pixels[i];
            tex.SetPixels32(outPixels);
            tex.Apply(updateMipmaps: false, makeNoLongerReadable: true);
            return tex;
        }

        private static readonly Dictionary<char, string[]> Glyphs = new Dictionary<char, string[]>
        {
            ['A'] = new[] { "01110", "10001", "10001", "11111", "10001", "10001", "10001" },
            ['I'] = new[] { "11111", "00100", "00100", "00100", "00100", "00100", "11111" },
            ['O'] = new[] { "01110", "10001", "10001", "10001", "10001", "10001", "01110" },
            ['F'] = new[] { "11111", "10000", "10000", "11110", "10000", "10000", "10000" },
            ['N'] = new[] { "10001", "11001", "10101", "10011", "10001", "10001", "10001" },
            ['E'] = new[] { "11111", "10000", "10000", "11110", "10000", "10000", "11111" },
            ['R'] = new[] { "11110", "10001", "10001", "11110", "10100", "10010", "10001" },
            ['·'] = new[] { "00000", "00000", "00100", "01110", "00100", "00000", "00000" },
            [' '] = new[] { "000", "000", "000", "000", "000", "000", "000" }
        };

        private static void DrawBitmapText(Color[] pixels, int texW, int texH, string text, int startX, int centerY, int scale, Color color)
        {
            int glyphH = 7 * scale;
            int baseY = centerY - glyphH / 2;
            int cursorX = startX;

            foreach (char ch in text)
            {
                if (!Glyphs.TryGetValue(ch, out string[] rows))
                {
                    cursorX += 3 * scale;
                    continue;
                }
                int cols = rows[0].Length;
                for (int r = 0; r < rows.Length; r++)
                {
                    int texRowBase = baseY + (rows.Length - 1 - r) * scale;
                    string rowStr = rows[r];
                    for (int c = 0; c < cols; c++)
                    {
                        if (rowStr[c] != '1') continue;
                        int texColBase = cursorX + c * scale;
                        for (int sy = 0; sy < scale; sy++)
                        {
                            int py = texRowBase + sy;
                            if (py < 0 || py >= texH) continue;
                            for (int sx = 0; sx < scale; sx++)
                            {
                                int px = texColBase + sx;
                                if (px < 0 || px >= texW) continue;
                                pixels[py * texW + px] = AlphaOver(color, pixels[py * texW + px]);
                            }
                        }
                    }
                }
                cursorX += (cols + 2) * scale;
            }
        }

        private static Texture2D CreateCircleTexture(int size)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, mipChain: false)
            {
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };
            var pixels = new Color32[size * size];
            float center = size * 0.5f;
            float radius = center - 2.0f;
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float dx = (x + 0.5f) - center;
                    float dy = (y + 0.5f) - center;
                    float dist = Mathf.Sqrt(dx * dx + dy * dy) - radius;
                    byte a = (byte)Mathf.RoundToInt(Mathf.Clamp01(0.6f - dist) * 255f);
                    pixels[y * size + x] = new Color32(255, 255, 255, a);
                }
            }
            tex.SetPixels32(pixels);
            tex.Apply(updateMipmaps: false, makeNoLongerReadable: true);
            return tex;
        }

        private static float RoundedBoxSdf(float px, float py, float halfW, float halfH, float radius)
        {
            float qx = Mathf.Abs(px) - (halfW - radius);
            float qy = Mathf.Abs(py) - (halfH - radius);
            float outDist = Mathf.Sqrt(Mathf.Max(qx, 0f) * Mathf.Max(qx, 0f) + Mathf.Max(qy, 0f) * Mathf.Max(qy, 0f));
            float inDist = Mathf.Min(Mathf.Max(qx, qy), 0f);
            return outDist + inDist - radius;
        }

        private static float SegmentDist(Vector2 p, Vector2 a, Vector2 b)
        {
            Vector2 ab = b - a;
            float lenSq = ab.sqrMagnitude;
            if (lenSq < 1e-5f) return Vector2.Distance(p, a);
            float t = Mathf.Clamp01(Vector2.Dot(p - a, ab) / lenSq);
            return Vector2.Distance(p, a + ab * t);
        }

        private static Color AlphaOver(Color fg, Color bg)
        {
            float a = fg.a + bg.a * (1f - fg.a);
            if (a < 1e-5f) return Color.clear;
            return new Color(
                (fg.r * fg.a + bg.r * bg.a * (1f - fg.a)) / a,
                (fg.g * fg.a + bg.g * bg.a * (1f - fg.a)) / a,
                (fg.b * fg.a + bg.b * bg.a * (1f - fg.a)) / a,
                a);
        }

        private static void DestroyTex(ref Texture2D tex)
        {
            if (tex != null)
            {
                UnityEngine.Object.Destroy(tex);
                tex = null;
            }
        }
    }
}
